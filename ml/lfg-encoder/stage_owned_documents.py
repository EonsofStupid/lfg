#!/usr/bin/env python3
"""Stage selected screened training documents for a private training host."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import corpus_documents
import owned_sft_checkpoint as checkpoints
import train_owned_student as base


def write_new(path: Path, data: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(data)


def stage(plan_path: Path, policy_path: Path, student_config: Path, output: Path) -> dict:
    config, config_sha = base.load_config(student_config)
    if output.exists():
        raise ValueError('staging output already exists')
    checkpoints.private_directory(output)
    corpus = corpus_documents.load(plan_path, policy_path, config['max_data_bytes'], config['architecture']['context'])
    code_paths = (Path(__file__), Path(corpus_documents.__file__), Path(base.__file__),
                  Path(base.screen_private_corpus.__file__), Path(checkpoints.__file__))
    code = {path.name: base.sha(path.read_bytes()) for path in code_paths}
    corpus_documents.verify(corpus)
    if base.load_config(student_config)[1] != config_sha:
        raise ValueError('student configuration changed during staging')
    output.mkdir(mode=0o700)
    staged_sources = []
    for index, (data, entry) in enumerate(zip(corpus['documents'], corpus['identity']['sources'], strict=True)):
        name = f'document-{index + 1:06d}.utf8'
        write_new(output / name, data)
        staged_sources.append({'path': name, 'project': entry['project'], 'project_group': entry['project_group'],
                               'split': 'train', 'sha256': base.sha(data)})
    staged_plan = {'schema_version': 'lfg.document-development-plan.v1',
                   'sampling': corpus['identity']['sampling'], 'sources': staged_sources}
    plan_raw = (json.dumps(staged_plan, sort_keys=True, indent=2, ensure_ascii=False) + '\n').encode('utf-8')
    write_new(output / 'plan.json', plan_raw)
    corpus_documents.verify(corpus)
    if (base.load_config(student_config)[1] != config_sha
            or any(base.sha(path.read_bytes()) != code[path.name] for path in code_paths)
            or any(base.sha((output / item['path']).read_bytes()) != item['sha256'] for item in staged_sources)):
        raise ValueError('staging inputs or outputs changed')
    manifest = {'schema_version': 'lfg.screened-document-stage.v1', 'source_corpus_sha256': corpus['corpus_sha256'],
                'source_identity': corpus['identity'], 'student_config_sha256': config_sha,
                'staged_plan_sha256': base.sha(plan_raw), 'code_sha256': code,
                'document_count': len(staged_sources), 'total_bytes': sum(len(data) for data in corpus['documents']),
                'validation_sources_opened': False, 'test_sources_opened': False,
                'redaction_complete': False, 'human_reviewed': False}
    write_new(output / 'manifest.json', (json.dumps(manifest, sort_keys=True, indent=2, ensure_ascii=False) + '\n').encode('utf-8'))
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'policy', 'student-config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = stage(args.plan, args.policy, args.student_config, args.output)
        print(json.dumps({key: result[key] for key in ('source_corpus_sha256', 'staged_plan_sha256', 'document_count',
              'total_bytes', 'validation_sources_opened', 'test_sources_opened', 'redaction_complete', 'human_reviewed')}))
    except (OSError, ValueError, TypeError, KeyError, UnicodeError, OverflowError, RuntimeError) as error:
        raise SystemExit(f'Owned document staging failed ({type(error).__name__}): {error}') from None
