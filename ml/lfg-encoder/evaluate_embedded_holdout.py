#!/usr/bin/env python3
"""Freeze an embedded candidate before comparing it on separate reviewed holdout inputs."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import sys

import evaluate_native_exemplars as native
import node_evaluate
import node_exemplars
import node_review


def holdout_commitment(inputs: list[list[Path]], snapshot: Path, node: str) -> dict:
    snapshot_sha = node_evaluate.hash_file(snapshot)
    sources, seen = [], set()
    for queue, ledger_path in inputs:
        items, ledger, raw = node_review.load_ledger(queue, snapshot, ledger_path)
        if (ledger['snapshotSha256'] != snapshot_sha or any(row['decision'] == 'pending' or
                (row['decision'] == 'approved' and row['split'] != 'test') for row in ledger['records'])):
            raise ValueError('holdout ledgers must be decided and contain only test approvals')
        target = {node_review.identity(item) for item in items if item['node'] == node}
        if not any(row['decision'] == 'approved' and row['itemSha256'] in target for row in ledger['records']):
            raise ValueError('holdout source has no approved target-node items')
        source = (ledger['queueSha256'], node_review.digest(raw))
        if source in seen:
            raise ValueError('duplicate holdout source')
        seen.add(source)
        if node_evaluate.hash_file(queue) != source[0] or node_evaluate.hash_file(ledger_path) != source[1]:
            raise ValueError('holdout source changed during commitment')
        sources.append({'queueSha256': source[0], 'ledgerSha256': source[1]})
    if not sources or node_evaluate.hash_file(snapshot) != snapshot_sha:
        raise ValueError('holdout sources are missing or snapshot changed')
    return {'schemaVersion': 'lfg.node-holdout-commitment.v1', 'node': node,
            'snapshotSha256': snapshot_sha,
            'sources': sorted(sources, key=lambda item: (item['queueSha256'], item['ledgerSha256']))}


def load_commitment(args: argparse.Namespace) -> tuple[dict, str]:
    raw = args.commitment.read_bytes()
    actual = node_review.digest(raw)
    if not node_review.HEX.fullmatch(args.commitment_sha256) or actual != args.commitment_sha256:
        raise ValueError('holdout commitment digest mismatch')
    value = json.loads(raw)
    if (not isinstance(value, dict) or set(value) != {'schemaVersion', 'node', 'snapshotSha256', 'sources'}
            or value['schemaVersion'] != 'lfg.node-holdout-commitment.v1' or value['node'] != args.node
            or value['snapshotSha256'] != node_evaluate.hash_file(args.snapshot)
            or not isinstance(value['sources'], list) or not value['sources']):
        raise ValueError('invalid holdout commitment')
    pairs = []
    for source in value['sources']:
        if (not isinstance(source, dict) or set(source) != {'queueSha256', 'ledgerSha256'}
                or any(not isinstance(source[key], str) or not node_review.HEX.fullmatch(source[key])
                       for key in ('queueSha256', 'ledgerSha256'))):
            raise ValueError('invalid holdout source commitment')
        pairs.append((source['queueSha256'], source['ledgerSha256']))
    if len(set(pairs)) != len(pairs) or pairs != sorted(pairs) or args.commitment.read_bytes() != raw:
        raise ValueError('holdout commitment is duplicated, unordered or changed')
    return value, actual


def candidate_predictions(args: argparse.Namespace, records: list[dict], baseline: dict,
                          audit: dict, options: set[str], runtime_sha: str, config_sha: str,
                          *, embedded: bool) -> dict:
    if not records or len(records) > args.max_items:
        raise ValueError('invalid evaluation item count')
    expected = {record['review']['itemSha256']: record for record in records}
    if len(expected) != len(records):
        raise ValueError('duplicate evaluation item')
    body = bytearray()
    for key, record in expected.items():
        line = node_review.encode({'id': key, 'text': record['item']['text']}) + b'\n'
        if len(body) + len(line) > args.max_input_bytes:
            raise ValueError('evaluation inputs exceed configured byte limit')
        body.extend(line)
    command = [str(args.binary.resolve(strict=True)),
               'embedded-node-suggest' if embedded else 'onnx-node-suggest', '--experimental',
               '--runtime-config', str(args.runtime_config), '--exemplar-config', str(args.exemplar_config)]
    if not embedded:
        command.extend(['--snapshot', str(args.snapshot), '--artifact', str(args.artifact),
                        '--manifest-sha256', args.manifest_sha256, '--exemplar-pack', str(args.pack),
                        '--pack-sha256', args.pack_sha256])
    process = subprocess.run(command, input=bytes(body), capture_output=True,
                             timeout=args.timeout_seconds, check=False)
    if process.returncode != 0:
        raise ValueError(f'embedded candidate failed with exit code {process.returncode}')
    if len(process.stdout) > args.max_output_bytes or not process.stdout.endswith(b'\n'):
        raise ValueError('embedded candidate output is incomplete or exceeds configured limit')
    result = {}
    fields = {'schemaVersion', 'id', 'inputSha256', 'snapshotSha256', 'encoderManifestSha256',
              'packSha256', 'exemplarConfigSha256', 'runtimeConfigSha256', 'baseline',
              'extensionStatus', 'suggestion', 'promotionEligible'}
    for line in process.stdout.splitlines():
        row = json.loads(line)
        if not isinstance(row, dict) or set(row) != fields or row['schemaVersion'] != 'lfg.node-suggestion-probe.v1':
            raise ValueError('invalid embedded suggestion record')
        key = row['id']
        if (key not in expected or key in result or row['inputSha256'] != expected[key]['item']['inputSha256']
                or row['snapshotSha256'] != audit['snapshotSha256'] or row['encoderManifestSha256'] != args.manifest_sha256
                or row['packSha256'] != args.pack_sha256 or row['runtimeConfigSha256'] != runtime_sha
                or row['exemplarConfigSha256'] != config_sha or row['promotionEligible'] is not False
                or row['baseline'] != baseline.get(key)):
            raise ValueError('embedded suggestion identity or baseline mismatch')
        state, _ = node_evaluate.baseline_choice(row['baseline'], args.node, options)
        suggestion = row['suggestion']
        if state != 'escalated':
            if row['extensionStatus'] != 'not_applicable' or suggestion is not None:
                raise ValueError('embedded candidate acted outside the target escalation')
            result[key] = None
        elif row['extensionStatus'] == 'abstained' and suggestion is None:
            result[key] = None
        elif row['extensionStatus'] == 'suggested' and isinstance(suggestion, dict):
            if (set(suggestion) != {'choice', 'similarity', 'margin', 'source'}
                    or suggestion['choice'] not in options or suggestion['source'] != 'exemplar'
                    or any(type(suggestion[k]) not in (int, float) or not math.isfinite(suggestion[k])
                           for k in ('similarity', 'margin'))
                    or not -1 <= suggestion['similarity'] <= 1 or not 0 <= suggestion['margin'] <= 2):
                raise ValueError('invalid embedded suggestion')
            result[key] = suggestion['choice']
        else:
            raise ValueError('invalid embedded escalation status')
    if set(result) != set(expected):
        raise ValueError('embedded suggestions do not cover evaluation items')
    return result


def embedded_predictions(args: argparse.Namespace, records: list[dict], baseline: dict,
                         audit: dict, options: set[str], runtime_sha: str, config_sha: str) -> dict:
    return candidate_predictions(args, records, baseline, audit, options, runtime_sha, config_sha, embedded=True)


def external_predictions(args: argparse.Namespace, records: list[dict], baseline: dict,
                         audit: dict, options: set[str], runtime_sha: str, config_sha: str) -> dict:
    return candidate_predictions(args, records, baseline, audit, options, runtime_sha, config_sha, embedded=False)


def freeze_candidate(args: argparse.Namespace) -> dict:
    if args.max_output_bytes < 1:
        raise ValueError('output byte limit must be positive')
    development_policy, _ = node_review.load_policy(args.policy)
    if development_policy['schemaVersion'] != 'lfg.node-admission-policy.v2':
        raise ValueError('candidate freeze requires a development-only policy')
    _, commitment_sha = load_commitment(args)
    policy, final_policy_sha = node_review.load_policy(args.final_policy)
    if policy['schemaVersion'] != 'lfg.node-admission-policy.v1':
        raise ValueError('final policy must include train, validation and test requirements')
    tool_sha = node_evaluate.hash_file(Path(__file__))
    # This validator uses development inputs only and checks the pack against
    # the current review audit before actually running the external candidate.
    development = native.run(args)
    audit = development['admission']
    if (audit.get('schemaVersion') != 'lfg.node-admission-report.v2'
            or audit.get('scope') != 'development' or audit.get('holdoutAssessment') != 'not_performed'):
        raise ValueError('candidate freeze requires development-only admission')
    current, records = node_review.audit(args.input, args.snapshot, args.policy, args.node)
    if current != audit:
        raise ValueError('development audit changed during freeze')
    validation = [record for record in records if record['review']['split'] == 'validation']
    options = set(json.loads(args.snapshot.read_bytes())['nodes'][args.node]['options'])
    baseline = node_evaluate.predictions(args.binary, args.snapshot, audit['snapshotSha256'], validation,
                                        args.timeout_seconds, args.max_items, args.max_input_bytes)
    candidate = embedded_predictions(args, validation, baseline, audit, options,
                                     development['runtimeConfigSha256'], development['exemplarConfigSha256'])
    external = external_predictions(args, validation, baseline, audit, options,
                                    development['runtimeConfigSha256'], development['exemplarConfigSha256'])
    if candidate != external:
        raise ValueError('embedded and external candidate validation choices differ')
    metrics = node_evaluate.score(validation, baseline, args.node, options, candidate)
    if metrics != development['metrics']:
        raise ValueError('embedded and external candidate validation metrics differ')
    if metrics['fallbackOnEscalation']['eligible'] == 0:
        raise ValueError('development validation has no target escalation to exercise the encoder')
    if (native.run(args) != development or load_commitment(args)[1] != commitment_sha
            or node_evaluate.hash_file(args.final_policy) != final_policy_sha
            or node_evaluate.hash_file(Path(__file__)) != tool_sha):
        raise ValueError('candidate inputs changed during freeze')
    return {'schemaVersion': 'lfg.embedded-node-freeze.v1', 'node': args.node,
            'developmentValidation': development, 'embeddedValidationMetrics': metrics,
            'finalPolicySha256': final_policy_sha, 'holdoutCommitmentSha256': commitment_sha,
            'maxOutputBytes': args.max_output_bytes,
            'evaluatorSha256': tool_sha, 'holdoutInputParametersAccepted': False,
            'qualificationEstablished': False}


def evaluate_holdout(args: argparse.Namespace, frozen: dict) -> dict:
    # Revalidate the frozen development candidate before opening holdout files.
    if freeze_candidate(args) != frozen:
        raise ValueError('current candidate differs from frozen development state')
    dev_paths = {path.resolve(strict=True) for pair in args.input for path in pair}
    holdout_paths = [path.resolve(strict=True) for pair in args.holdout_input for path in pair]
    if len(set(holdout_paths)) != len(holdout_paths) or dev_paths.intersection(holdout_paths):
        raise ValueError('development and holdout files must be distinct')
    expected_commitment, _ = load_commitment(args)
    if holdout_commitment(args.holdout_input, args.snapshot, args.node) != expected_commitment:
        raise ValueError('holdout inputs differ from the precommitted sources')
    combined, records = node_review.audit(args.input + args.holdout_input, args.snapshot, args.final_policy, args.node)
    if not combined['admissionEligible']:
        raise ValueError('combined holdout separation or support gate failed')
    holdout = [record for record in records if record['review']['split'] == 'test']
    options = set(json.loads(args.snapshot.read_bytes())['nodes'][args.node]['options'])
    baseline = node_evaluate.predictions(args.binary, args.snapshot, combined['snapshotSha256'], holdout,
                                        args.timeout_seconds, args.max_items, args.max_input_bytes)
    dev = frozen['developmentValidation']
    candidate = embedded_predictions(args, holdout, baseline, combined, options,
                                     dev['runtimeConfigSha256'], dev['exemplarConfigSha256'])
    metrics = node_evaluate.score(holdout, baseline, args.node, options, candidate)
    if metrics['fallbackOnEscalation']['eligible'] == 0:
        raise ValueError('holdout has no target escalation to evaluate the candidate')
    after, _ = node_review.audit(args.input + args.holdout_input, args.snapshot, args.final_policy, args.node)
    if after != combined or freeze_candidate(args) != frozen:
        raise ValueError('candidate or review inputs changed during holdout evaluation')
    return {'schemaVersion': 'lfg.embedded-node-holdout-report.v1', 'split': 'test',
            'node': args.node, 'freezeSha256': args.freeze_sha256, 'combinedAdmission': combined,
            'candidateProcessObserved': True, 'candidateAuthenticated': False,
            'qualificationEstablished': False, 'metrics': metrics}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    commit = sub.add_parser('commit-holdout', help='custodian commits reviewed test-file identities')
    commit.add_argument('--holdout-input', type=Path, nargs=2, action='append', required=True,
                        metavar=('TEST_QUEUE', 'TEST_LEDGER'))
    commit.add_argument('--snapshot', type=Path, required=True)
    commit.add_argument('--node', required=True)
    commit.add_argument('--output', type=Path, required=True)
    for name in ('freeze', 'test'):
        command = sub.add_parser(name)
        command.add_argument('--input', type=Path, nargs=2, action='append', required=True, metavar=('DEV_QUEUE', 'DEV_LEDGER'))
        for field in ('snapshot', 'policy', 'final-policy', 'binary', 'artifact', 'runtime-config', 'pack', 'exemplar-config', 'output'):
            command.add_argument('--' + field, type=Path, required=True)
        for field in ('node', 'manifest-sha256', 'pack-sha256'):
            command.add_argument('--' + field, required=True)
        command.add_argument('--commitment', type=Path, required=True)
        command.add_argument('--commitment-sha256', required=True)
        for field in ('max-items', 'max-input-bytes', 'max-output-bytes'):
            command.add_argument('--' + field, type=int, required=True)
        command.add_argument('--timeout-seconds', type=float, required=True)
        if name == 'test':
            command.add_argument('--freeze', type=Path, required=True)
            command.add_argument('--freeze-sha256', required=True)
            command.add_argument('--holdout-input', type=Path, nargs=2, action='append', required=True,
                                 metavar=('TEST_QUEUE', 'TEST_LEDGER'))
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('output already exists')
    if args.command == 'commit-holdout':
        result = holdout_commitment(args.holdout_input, args.snapshot, args.node)
    elif args.command == 'freeze':
        result = freeze_candidate(args)
    else:
        raw = args.freeze.read_bytes()
        if not node_review.HEX.fullmatch(args.freeze_sha256) or node_review.digest(raw) != args.freeze_sha256:
            raise ValueError('freeze digest mismatch')
        frozen = json.loads(raw)
        if not isinstance(frozen, dict) or frozen.get('schemaVersion') != 'lfg.embedded-node-freeze.v1':
            raise ValueError('invalid frozen candidate')
        result = evaluate_holdout(args, frozen)
        if args.freeze.read_bytes() != raw:
            raise ValueError('freeze changed during holdout evaluation')
    raw = node_review.encode(result) + b'\n'
    node_review.create_new(args.output, raw)
    print(json.dumps({'schemaVersion': 'lfg.embedded-node-evaluation-output.v1',
                      'outputSha256': node_review.digest(raw), 'qualificationEstablished': False}))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired, json.JSONDecodeError) as error:
        print(f'embedded node evaluation: {error}', file=sys.stderr)
        sys.exit(1)
