#!/usr/bin/env python3
"""Compare a native shadow artifact and deterministic baseline on an audited test set."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
from pathlib import Path
from typing import Any

import benchmark_persistent
import calibration
import corpus_audit
import evaluate
import predict_model
import train


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def audited_test(plan_path: Path, training_dataset: Path, run_manifest: dict[str, Any]) -> tuple[dict, list[dict], dict]:
    training, dataset_hash = train.load_dataset(training_dataset)
    report, validation = corpus_audit.load_reviewed_validation(training, plan_path)
    environment = run_manifest.get('environment_evidence')
    model_input = run_manifest.get('model_snapshot_evidence')
    digest = lambda value: isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None
    if (run_manifest.get('schema_version') != 'lfg.encoder-run.v1'
            or run_manifest.get('development_override') is not False
            or run_manifest.get('dataset_sha256') != dataset_hash
            or not isinstance(environment, dict) or not digest(environment.get('lock_sha256'))
            or not digest(environment.get('installed_versions_sha256'))
            or not isinstance(model_input, dict) or not digest(model_input.get('manifest_sha256'))
            or not digest(run_manifest.get('training_inputs_sha256'))
            or not digest(run_manifest.get('wheelhouse_verifier_sha256'))):
        raise ValueError('model run is not bound to the reviewed training input')
    split = run_manifest.get('split')
    fields = ('schema_version', 'plan_sha256', 'taxonomy_sha256', 'review_ledger_sha256',
              'auditor_sha256', 'taxonomy_loader_sha256', 'sources', 'counts', 'duplicate_within_split')
    if (not isinstance(split, dict) or split.get('development_override') is not False
            or split.get('validation_source') != 'audited-validation-partition'
            or split.get('dataset_sha256') != dataset_hash
            or split.get('corpus_audit') != {key: report[key] for key in fields}):
        raise ValueError('model run corpus audit differs from the reviewed plan')
    for field, rows, column in (('train_ids', training, 'id'), ('validation_ids', validation, 'id'),
                                ('train_families', training, 'family'), ('validation_families', validation, 'family')):
        expected = {row[column] for row in rows}
        actual = split.get(field)
        if (not isinstance(actual, list) or any(not isinstance(item, str) for item in actual)
                or len(actual) != len(expected) or set(actual) != expected):
            raise ValueError('model run split differs from the reviewed plan')
    raw_plan = plan_path.read_bytes()
    if hashlib.sha256(raw_plan).hexdigest() != report['plan_sha256']:
        raise ValueError('plan changed during test loading')
    plan = json.loads(raw_plan)
    test = []
    for source, evidence in zip(plan['sources'], report['sources'], strict=True):
        if source['split'] == 'test':
            rows, digest = train.load_dataset(corpus_audit.resolve(plan_path.parent, source['path']))
            if digest != evidence['sha256']:
                raise ValueError('test source changed during loading')
            test.extend(rows)
    if not test or corpus_audit.validate_training_corpus(training, plan_path) != report:
        raise ValueError('reviewed test partition is empty or audit changed')
    taxonomy_path = corpus_audit.resolve(plan_path.parent, plan['taxonomy'])
    taxonomy, taxonomy_hash = evaluate.load_taxonomy(taxonomy_path)
    if taxonomy_hash != report['taxonomy_sha256']:
        raise ValueError('evaluation taxonomy changed')
    return report, test, taxonomy


def checked_artifact(directory: Path, source_digest: str, taxonomy_digest: str) -> tuple[dict, str]:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError('ONNX artifact must be a directory')
    manifest_path = directory / 'manifest.json'
    if manifest_path.is_symlink():
        raise ValueError('ONNX manifest cannot be a link')
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if (not isinstance(manifest, dict) or manifest.get('schema_version') != 'lfg.encoder-onnx-artifact.v2'
            or manifest.get('source_artifact_sha256') != source_digest
            or manifest.get('taxonomy_sha256') != taxonomy_digest
            or not isinstance(manifest.get('files_sha256'), dict) or not manifest['files_sha256']):
        raise ValueError('ONNX artifact is not bound to the selected model and taxonomy')
    files = manifest['files_sha256']
    for name, expected in files.items():
        if (not isinstance(name, str) or not name or name in ('.', '..', 'manifest.json')
                or '/' in name or '\\' in name or not isinstance(expected, str)):
            raise ValueError('invalid ONNX artifact file entry')
        path = directory / name
        if path.is_symlink() or not path.is_file() or file_hash(path) != expected:
            raise ValueError('ONNX artifact file is missing or changed')
    if manifest_path.read_bytes() != raw:
        raise ValueError('ONNX manifest changed during verification')
    return manifest, hashlib.sha256(raw).hexdigest()


def compare(plan_path: Path, training_dataset: Path, model_dir: Path, artifact_dir: Path,
            binary: Path, runtime_config: Path, benchmark_config: Path, mode: str,
            calibration_config: Path | None = None) -> dict:
    if mode not in ('embedded', 'external'):
        raise ValueError('unsupported native artifact mode')
    config, config_hash = benchmark_persistent.load_config(benchmark_config)
    code_paths = [Path(__file__), Path(benchmark_persistent.__file__), Path(corpus_audit.__file__),
                  Path(evaluate.__file__), Path(predict_model.__file__), Path(train.__file__), Path(calibration.__file__)]
    code_hashes = {path.name: file_hash(path) for path in code_paths}
    model_hash = predict_model.artifact_digest(model_dir)
    manifest = json.loads((model_dir / 'run-manifest.json').read_bytes())
    audit_report, rows, taxonomy = audited_test(plan_path, training_dataset, manifest)
    if len(rows) > config['max_records'] or sum(len(row['text'].encode('utf-8')) for row in rows) > config['max_dataset_bytes']:
        raise ValueError('reviewed test exceeds configured benchmark input limits')
    label_file = json.loads((model_dir / 'labels.json').read_bytes())
    if not isinstance(label_file, list) or len(label_file) != len(set(label_file)) or set(label_file) != set.union(*taxonomy.values()):
        raise ValueError('trained model does not cover the complete evaluation taxonomy')
    onnx_manifest, onnx_hash = checked_artifact(artifact_dir, model_hash, audit_report['taxonomy_sha256'])
    if (onnx_manifest['files_sha256'].get('labels.json') != file_hash(model_dir / 'labels.json')
            or onnx_manifest['files_sha256'].get('taxonomy.json') != audit_report['taxonomy_sha256']):
        raise ValueError('ONNX label order or taxonomy differs from the training artifacts')
    edges = calibration_hash = decision = None
    if calibration_config is not None:
        edges, calibration_hash = calibration.load_config(calibration_config)
        decision, decision_hash = predict_model.prediction_config(artifact_dir / 'prediction-config.json')
        if (decision_hash != onnx_manifest.get('prediction_config_sha256')
                or set(decision['label_thresholds']) - set.union(*taxonomy.values())):
            raise ValueError('prediction thresholds differ from the native artifact or taxonomy')
    binary = binary.resolve(strict=True)
    binary_hash = file_hash(binary)
    runtime_hash = file_hash(runtime_config)
    overrides = {key: Path(os.environ[key]) for key in ('LFG_PREFLIGHT_CONFIG', 'LFG_OPERATIONS_CONFIG') if key in os.environ}
    override_hashes = {key: file_hash(path) for key, path in overrides.items()}
    arguments = (['embedded-onnx-predict', '--runtime-config', str(runtime_config)] if mode == 'embedded'
                 else ['onnx-predict', '--artifact', str(artifact_dir), '--manifest-sha256', onnx_hash,
                       '--runtime-config', str(runtime_config)])
    if calibration_config is not None:
        arguments.append('--include-scores')
    processes = {}
    try:
        for name, args in (('baseline', ['baseline-predict']), ('candidate', arguments)):
            processes[name] = benchmark_persistent.Predictor(binary, args, config)
        predictions = {'baseline': {}, 'candidate': {}}
        for index, row in enumerate(rows):
            for name in (('baseline', 'candidate') if index % 2 == 0 else ('candidate', 'baseline')):
                _, value = processes[name].request_prediction(row['id'], row['text'])
                predictions[name][row['id']] = value
        baseline, candidate = processes['baseline'], processes['candidate']
        if any(predictor.process.poll() is not None for predictor in processes.values()):
            raise ValueError('prediction process exited during evaluation')
        if baseline.producer != binary_hash or candidate.producer != binary_hash:
            raise ValueError('prediction producer differs from the selected binary')
        if (candidate.model.get('artifact') != onnx_hash or candidate.model.get('runtime') != binary_hash
                or candidate.model.get('taxonomy') != audit_report['taxonomy_sha256']
                or candidate.model.get('config') != runtime_hash):
            raise ValueError('native prediction identity differs from the selected inputs')
        identity = {key: candidate.model[key] for key in ('artifact', 'config', 'taxonomy', 'runtime')}
        encoded = json.dumps(identity, sort_keys=True, separators=(',', ':')).encode('utf-8')
        if candidate.model.get('sha256') != hashlib.sha256(encoded).hexdigest():
            raise ValueError('native prediction identity digest is invalid')
        scores = {name: evaluate.score(rows, value, taxonomy) for name, value in predictions.items()}
        calibrated_report = (calibration.calculate(rows, predictions['candidate'], taxonomy, decision, edges)
                             if calibration_config is not None else None)
        support = {field: {label: {'positive': scores['candidate'][field]['labels'][label]['support'],
                                   'negative': len(rows) - scores['candidate'][field]['labels'][label]['support']}
                           for label in sorted(labels)} for field, labels in taxonomy.items()}
        if (file_hash(binary) != binary_hash or file_hash(runtime_config) != runtime_hash
                or file_hash(benchmark_config) != config_hash or predict_model.artifact_digest(model_dir) != model_hash
                or checked_artifact(artifact_dir, model_hash, audit_report['taxonomy_sha256'])[1] != onnx_hash
                or {key: file_hash(path) for key, path in overrides.items()} != override_hashes
                or {path.name: file_hash(path) for path in code_paths} != code_hashes
                or (calibration_config is not None and file_hash(calibration_config) != calibration_hash)
                or audited_test(plan_path, training_dataset, manifest)[0] != audit_report):
            raise ValueError('evaluation inputs changed during inference')
        return {'schema_version': 'lfg.reviewed-native-evaluation.v1', 'promotion_eligible': False,
                'reason': 'promotion policy, performance qualification, and independent review remain separate',
                'examples': len(rows), 'audit': audit_report, 'binary_sha256': binary_hash,
                'source_artifact_sha256': model_hash, 'onnx_manifest_sha256': onnx_hash,
                'runtime_config_sha256': runtime_hash, 'evaluation_config_sha256': config_hash,
                'training_dataset_sha256': manifest['dataset_sha256'], 'code_sha256': code_hashes,
                'overrides_sha256': override_hashes, 'mode': mode,
                'platform': platform.platform(), 'machine': platform.machine(),
                'python_version': platform.python_version(), 'logical_cpus': os.cpu_count(),
                'models': {'baseline': baseline.model, 'candidate': candidate.model}, 'groups': scores,
                'test_label_support': support,
                'calibration_config_sha256': calibration_hash,
                'calibration': calibrated_report, 'performance_qualification': None}
    finally:
        for predictor in processes.values():
            predictor.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'training-dataset', 'model', 'artifact', 'binary', 'runtime-config', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--mode', choices=('embedded', 'external'), required=True)
    parser.add_argument('--calibration-config', type=Path, help='optional explicit score-bin edges for uncalibrated ECE')
    args = parser.parse_args()
    try:
        result = compare(args.plan, args.training_dataset, args.model, args.artifact, args.binary,
                         args.runtime_config, args.config, args.mode, args.calibration_config)
        with args.output.open('x', encoding='utf-8') as output:
            json.dump(result, output, indent=2, sort_keys=True, allow_nan=False)
            output.write('\n')
        print(json.dumps({'examples': result['examples'], 'promotion_eligible': False}))
    except (OSError, ValueError, KeyError, TypeError, UnicodeError) as error:
        raise SystemExit(f'reviewed native evaluation rejected: {error}') from error


if __name__ == '__main__':
    main()
