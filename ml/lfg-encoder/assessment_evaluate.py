#!/usr/bin/env python3
"""Compare supplied assessment predictions on reviewed development validation data."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re

import assessment_targets
import corpus_audit
import train

HEX = re.compile(r'^[0-9a-f]{64}$')
FIELDS = ('complexity', 'models', 'agent_count')


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def file_hash(path: Path) -> str:
    return sha(path.read_bytes())


def validation_records(plan_path: Path) -> tuple[dict, list[dict]]:
    """Check the plan scope before opening any dataset, and require current review."""
    raw = plan_path.read_bytes()
    plan = json.loads(raw)
    sources = plan.get('sources') if isinstance(plan, dict) else None
    if (not isinstance(plan, dict) or plan.get('schema_version') != corpus_audit.DEVELOPMENT_PLAN_SCHEMA
            or not isinstance(sources, list) or not sources
            or any(not isinstance(source, dict) or source.get('split') not in ('train', 'validation') for source in sources)
            or {source['split'] for source in sources} != {'train', 'validation'}):
        raise ValueError('assessment validation requires a train-and-validation-only development plan')
    audit, _ = corpus_audit.audit(plan_path, development_only=True)
    if sha(raw) != audit['plan_sha256']:
        raise ValueError('plan changed after scope check')
    if audit['duplicate_within_split']:
        raise ValueError('assessment validation requires deduplicated splits')
    for split in ('train', 'validation'):
        counts = audit['counts'][split]
        if (not counts.get('examples') or counts.get('labels_pending', 0)
                or counts.get('ready_for_reviewed_subset', 0) != counts['examples']):
            raise ValueError('every train and validation record requires a current review attestation')
    ledger_path = corpus_audit.resolve(plan_path.parent, plan.get('review_ledger'))
    ledger, ledger_sha = corpus_audit.read_ledger(ledger_path)
    if ledger_sha != audit['review_ledger_sha256']:
        raise ValueError('review ledger changed after audit')
    records = []
    for source in sources:
        if source['split'] != 'validation':
            continue
        rows, _ = train.load_dataset(corpus_audit.resolve(plan_path.parent, source['path']))
        for row in rows:
            annotation = corpus_audit.annotation_digest(row, source['project'], 'validation', audit['taxonomy_sha256'])
            if not row['verified'] or not corpus_audit.approved(ledger.get(row['id']), annotation):
                raise ValueError('every validation record must have a current review attestation')
            records.append({'row': row, 'annotation_sha256': annotation})
    if not records:
        raise ValueError('validation split is empty')
    if not any(record['row'].get('assessment_targets') and any(
            record['row']['assessment_targets']['values'][field] is not None for field in FIELDS) for record in records):
        raise ValueError('reviewed validation has no known assessment targets')
    return audit, records


def read_predictions(path: Path, records: list[dict], max_bytes: int) -> tuple[dict, dict]:
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError('prediction byte limit must be positive')
    with path.open('rb') as stream:
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes or not raw.endswith(b'\n'):
        raise ValueError('prediction file exceeds its limit or is incomplete')
    expected = {record['row']['id']: record for record in records}
    if len(expected) != len(records):
        raise ValueError('duplicate validation record')
    result = {}
    identity = None
    keys = {'schema_version', 'id', 'input_sha256', 'annotation_sha256', 'profile_sha256',
            'producer_sha256', 'artifact_sha256', 'values'}
    for line in raw.splitlines():
        item = json.loads(line)
        if (not isinstance(item, dict) or set(item) != keys or item['schema_version'] != 'lfg.assessment-prediction.v1'
                or not isinstance(item['id'], str) or item['id'] not in expected or item['id'] in result
                or any(not isinstance(item[name], str) or not HEX.fullmatch(item[name])
                       for name in ('input_sha256', 'annotation_sha256', 'producer_sha256', 'artifact_sha256'))):
            raise ValueError('invalid assessment prediction identity')
        record = expected[item['id']]
        row = record['row']
        targets = row.get('assessment_targets')
        profile_sha = targets['profile_sha256'] if targets else None
        if (item['input_sha256'] != sha(row['text'].encode('utf-8'))
                or item['annotation_sha256'] != record['annotation_sha256'] or item['profile_sha256'] != profile_sha):
            raise ValueError('prediction does not match the reviewed input and project profile')
        values = item['values']
        if targets:
            assessment_targets.make_record(values, targets['project_profile'], targets['config_revision'], targets['config_sha256'])
        elif not isinstance(values, dict) or set(values) != set(FIELDS) or any(values[field] is not None for field in FIELDS):
            raise ValueError('a record without a reviewed profile must have unknown predictions')
        current = {'producer_sha256': item['producer_sha256'], 'artifact_sha256': item['artifact_sha256']}
        if identity is not None and current != identity:
            raise ValueError('prediction file contains multiple producer identities')
        identity = current
        result[item['id']] = values
    if set(result) != set(expected):
        raise ValueError('predictions must cover the complete validation split')
    with path.open('rb') as stream:
        if stream.read(max_bytes + 1) != raw:
            raise ValueError('prediction file changed while reading')
    return result, {'file_sha256': sha(raw), **identity}


def ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def score(records: list[dict], predictions: dict) -> dict:
    counts = {'complexity': {'known': 0, 'unknown': 0, 'predicted': 0, 'correct': 0},
              'agent_count': {'known': 0, 'unknown': 0, 'predicted': 0, 'exact': 0, 'absolute_error_sum': 0},
              'models': {'known': 0, 'unknown': 0, 'predicted': 0, 'exact': 0, 'expected_models': 0, 'tp': 0, 'fp': 0, 'fn': 0},
              'effort': {'known': 0, 'unknown_for_expected_models': 0, 'models_unknown_records': 0, 'predicted': 0, 'correct': 0}}
    for record in records:
        row = record['row']
        actual = row['assessment_targets']['values'] if row.get('assessment_targets') else {field: None for field in FIELDS}
        predicted = predictions[row['id']]
        for field in FIELDS:
            counts[field]['known' if actual[field] is not None else 'unknown'] += 1
            if actual[field] is not None and predicted[field] is not None:
                counts[field]['predicted'] += 1
        if actual['complexity'] is not None and predicted['complexity'] is not None:
            counts['complexity']['correct'] += int(actual['complexity'] == predicted['complexity'])
        if actual['agent_count'] is not None and predicted['agent_count'] is not None:
            difference = abs(actual['agent_count'] - predicted['agent_count'])
            counts['agent_count']['absolute_error_sum'] += difference
            counts['agent_count']['exact'] += int(difference == 0)
        actual_models = {model['id']: model['effort'] for model in actual['models']} if actual['models'] is not None else None
        predicted_models = {model['id']: model['effort'] for model in predicted['models']} if predicted['models'] is not None else None
        if actual_models is not None:
            counts['models']['expected_models'] += len(actual_models)
        else:
            counts['effort']['models_unknown_records'] += 1
        if actual_models is not None and predicted_models is not None:
            expected, supplied = set(actual_models), set(predicted_models)
            counts['models']['exact'] += int(expected == supplied)
            counts['models']['tp'] += len(expected & supplied)
            counts['models']['fp'] += len(supplied - expected)
            counts['models']['fn'] += len(expected - supplied)
        if actual_models is not None:
            for model, effort in actual_models.items():
                if effort is None:
                    counts['effort']['unknown_for_expected_models'] += 1
                    continue
                counts['effort']['known'] += 1
                prediction = None if predicted_models is None else predicted_models.get(model)
                if prediction is not None:
                    counts['effort']['predicted'] += 1
                    counts['effort']['correct'] += int(effort == prediction)
    for field in FIELDS:
        counts[field]['coverage'] = ratio(counts[field]['predicted'], counts[field]['known'])
    counts['complexity']['accuracy_among_predicted'] = ratio(counts['complexity']['correct'], counts['complexity']['predicted'])
    counts['agent_count']['mean_absolute_error'] = ratio(counts['agent_count']['absolute_error_sum'], counts['agent_count']['predicted'])
    counts['models']['precision_among_predicted'] = ratio(counts['models']['tp'], counts['models']['tp'] + counts['models']['fp'])
    counts['models']['recall_among_predicted'] = ratio(counts['models']['tp'], counts['models']['tp'] + counts['models']['fn'])
    counts['models']['recall_including_abstentions'] = ratio(counts['models']['tp'], counts['models']['expected_models'])
    counts['effort']['coverage'] = ratio(counts['effort']['predicted'], counts['effort']['known'])
    counts['effort']['accuracy_among_predicted'] = ratio(counts['effort']['correct'], counts['effort']['predicted'])
    return counts


def evaluate(plan: Path, baseline: Path, candidate: Path, max_prediction_bytes: int) -> dict:
    code = [Path(__file__), Path(assessment_targets.__file__), Path(corpus_audit.__file__), Path(train.__file__),
            Path(corpus_audit.evaluate.__file__)]
    code_hashes = {path.name: file_hash(path) for path in code}
    audit, records = validation_records(plan)
    baseline_values, baseline_identity = read_predictions(baseline, records, max_prediction_bytes)
    candidate_values, candidate_identity = read_predictions(candidate, records, max_prediction_bytes)
    metrics = {'baseline': score(records, baseline_values), 'candidate': score(records, candidate_values)}
    audit_after, _ = validation_records(plan)
    if (audit_after != audit or read_predictions(baseline, records, max_prediction_bytes)[1] != baseline_identity
            or read_predictions(candidate, records, max_prediction_bytes)[1] != candidate_identity
            or any(file_hash(path) != code_hashes[path.name] for path in code)):
        raise ValueError('evaluation inputs changed while scoring')
    return {'schema_version': 'lfg.assessment-validation-report.v1', 'split': 'validation', 'audit': audit,
            'code_sha256': code_hashes, 'baseline': baseline_identity, 'candidate': candidate_identity,
            'examples': len(records), 'metrics': metrics, 'prediction_execution_verified': False,
            'qualification_established': False,
            'notice': 'Supplied predictions and reviewer identities are claims; unknowns are not negatives.'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'baseline', 'candidate', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--max-prediction-bytes', type=int, required=True)
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError('output already exists')
        report = evaluate(args.plan, args.baseline, args.candidate, args.max_prediction_bytes)
        raw = (json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(raw)
        print(json.dumps({'examples': report['examples'], 'output_sha256': sha(raw), 'qualification_established': False}))
    except (OSError, ValueError, KeyError, TypeError, UnicodeError) as error:
        raise SystemExit(f'assessment validation rejected: {error}') from error


if __name__ == '__main__':
    main()
