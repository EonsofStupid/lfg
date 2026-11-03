#!/usr/bin/env python3
"""Preserve a completed record prefix and isolate one uncollectible request."""
from collections import Counter
import argparse
import json
import os
from pathlib import Path
import shutil

import collect_local_teacher as collector
import screen_private_corpus as screen
import train
from owned_sft_checkpoint import private_directory


def split(selection: Path, config_path: Path, policy_path: Path, original: Path, destination: Path) -> dict:
    config, config_raw = collector.config_file(config_path)
    _, _, policy_sha = screen.load_policy(policy_path, config['max_manifest_bytes'])
    selection_raw = (selection / 'manifest.json').read_bytes()
    source_raw = (selection / 'requests.jsonl').read_bytes()
    manifest = json.loads(selection_raw)
    rows = [json.loads(line) for line in source_raw.splitlines()]
    identity_raw = (original / 'run.json').read_bytes()
    expected = {'schema_version': 'lfg.local-teacher-run.v1', 'config_sha256': collector.sha(config_raw),
                'selection_manifest_sha256': collector.sha(selection_raw), 'requests_sha256': collector.sha(source_raw),
                'policy_sha256': policy_sha, 'collector_sha256': collector.sha(Path(collector.__file__).read_bytes()),
                'screen_code_sha256': collector.sha(Path(screen.__file__).read_bytes())}
    if (identity_raw != collector.encoded(expected) or manifest.get('requests_sha256') != collector.sha(source_raw)
            or manifest.get('requests') != len(rows) or manifest.get('validation_opened') is not False
            or manifest.get('test_opened') is not False):
        raise ValueError('original collection identity mismatch')
    private_directory(destination)
    if destination.exists():
        raise ValueError('destination already exists')
    with train.training_run_lock(original):
        records = original / 'records'
        names = sorted(p.name for p in records.glob('*.json'))
        count = len(names)
        if count < 1 or count >= len(rows) or names != [f'{i:06d}.json' for i in range(count)]:
            raise ValueError('a contiguous record prefix and a remaining request are required')
        destination.mkdir(mode=0o700)
        def write_selection(name, selected):
            folder = destination / name
            folder.mkdir(mode=0o700)
            raw = b''.join(collector.encoded(row) for row in selected)
            derived = {**manifest, 'requests': len(selected), 'requests_sha256': collector.sha(raw),
                       'projects': dict(sorted(Counter(row['project'] for row in selected).items())),
                       'derived_from_selection_sha256': collector.sha(selection_raw),
                       'excluded_incomplete_request_id': rows[count]['id'],
                       'redaction_counts': {}, 'redaction_counts_recomputed': False}
            collector.write_new(folder / 'requests.jsonl', raw)
            collector.write_new(folder / 'manifest.json', collector.encoded(derived))
            return collector.encoded(derived), raw
        prefix_manifest, prefix_raw = write_selection('prefix-selection', rows[:count])
        write_selection('remaining-selection', rows[count + 1:])
        output = destination / 'prefix-output'
        output.mkdir(mode=0o700)
        (output / 'records').mkdir(mode=0o700)
        for name in names:
            path = records / name
            if path.is_symlink() or not path.is_file():
                raise ValueError('original record must be a regular file')
            shutil.copyfile(path, output / 'records' / name)
            os.chmod(output / 'records' / name, 0o600)
        derived_identity = {**expected, 'selection_manifest_sha256': collector.sha(prefix_manifest),
                            'requests_sha256': collector.sha(prefix_raw)}
        collector.write_new(output / 'run.json', collector.encoded(derived_identity))
        report = {'schema_version': 'lfg.local-teacher-split.v1', 'original_run_sha256': collector.sha(identity_raw),
                  'original_selection_sha256': collector.sha(selection_raw), 'completed_prefix': count,
                  'excluded_request_id': rows[count]['id'], 'excluded_reason': 'incomplete_teacher_response',
                  'remaining': len(rows) - count - 1,
                  'records_require_collector_resume_validation': True}
        collector.write_new(destination / 'split.json', collector.encoded(report))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('selection', 'config', 'policy', 'original', 'destination'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = split(args.selection, args.config, args.policy, args.original, args.destination)
    print(json.dumps({k: result[k] for k in ('completed_prefix', 'excluded_reason', 'remaining')}))


if __name__ == '__main__':
    main()
