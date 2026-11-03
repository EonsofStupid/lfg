#!/usr/bin/env python3
"""Build screened teacher requests from verified train-only conversation context."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import inspect_claude_inventory as claude
import inspect_codex_inventory as codex
import corpus_audit
import inventory_claude_sessions
import owned_sft_checkpoint
import screen_private_corpus as screen
import select_teacher_requests
from owned_sft_checkpoint import private_directory
from select_teacher_requests import encoded, sha


def select(plan_path: Path, dataset: Path, policy_path: Path, config_path: Path, output: Path) -> dict:
    code_raw = Path(__file__).read_bytes()
    config_raw = screen._read(config_path, 1048576, 'config_unavailable')
    cfg = json.loads(config_raw)
    fields = ('max_manifest_bytes', 'max_train_bytes', 'max_source_bytes', 'max_session_bytes',
              'max_line_bytes', 'max_prior_messages', 'max_prior_bytes', 'max_ancestry_records',
              'min_followup_bytes', 'max_followup_bytes', 'max_request_bytes', 'max_candidates', 'max_selected', 'max_per_project')
    if (not isinstance(cfg, dict) or cfg.get('schema_version') != 'lfg.contextual-teacher-selection.v1'
            or any(type(cfg.get(k)) is not int or cfg[k] < 1 for k in fields)
            or any(not isinstance(cfg.get(k), str) or not cfg[k] for k in ('seed', 'prior_prefix', 'separator', 'current_prefix'))
            or not isinstance(cfg.get('exclude_ids', []), list)
            or any(not isinstance(x, str) or not x for x in cfg.get('exclude_ids', []))
            or not isinstance(cfg.get('exclude_normalized_sha256', []), list)
            or any(not isinstance(x, str) or len(x) != 64 or any(c not in '0123456789abcdef' for c in x)
                   for x in cfg.get('exclude_normalized_sha256', []))
            or cfg['min_followup_bytes'] > cfg['max_followup_bytes']):
        raise ValueError('invalid contextual selection configuration')
    excluded = set(cfg.get('exclude_ids', []))
    policy, rules, policy_sha = screen.load_policy(policy_path, cfg['max_manifest_bytes'])
    helpers = (codex, claude, inventory_claude_sessions, owned_sft_checkpoint, screen, corpus_audit, select_teacher_requests)
    helper_hashes = {Path(module.__file__).name: sha(Path(module.__file__).read_bytes()) for module in helpers}
    plan_raw = screen._read(plan_path, cfg['max_manifest_bytes'], 'plan_unavailable')
    plan = json.loads(plan_raw)
    if (not isinstance(plan, dict) or plan.get('schema_version') != 'lfg.encoder-development-corpus-plan.v2'
            or not isinstance(plan.get('sources'), list)):
        raise ValueError('invalid source plan')
    manifest_raw = screen._read(dataset / 'manifest.json', cfg['max_manifest_bytes'], 'manifest_unavailable')
    manifest = json.loads(manifest_raw)
    train_raw = screen._read(dataset / 'train.jsonl', cfg['max_train_bytes'], 'train_unavailable')
    if (not isinstance(manifest, dict) or manifest.get('schema_version') != 'lfg.private-corpus-screen-manifest.v1'
            or manifest.get('plan_sha256') != sha(plan_raw) or not isinstance(manifest.get('policy_sha256'), str)
            or manifest.get('outputs', {}).get('train') != sha(train_raw)):
        raise ValueError('screened training identity does not match source plan')
    rows = [json.loads(line) for line in train_raw.splitlines()]
    if not rows or len(rows) > cfg['max_candidates']:
        raise ValueError('invalid candidate count')
    candidates = {}
    for row in rows:
        if (not isinstance(row, dict) or row.get('schema_version') != screen.OUTPUT_VERSION
                or row.get('split') != 'train' or row.get('review_status') != 'unreviewed'
                or not isinstance(row.get('id'), str) or row['id'] in candidates
                or not isinstance(row.get('text'), str) or sha(row['text'].encode()) != row.get('screened_text_sha256')
                or row.get('screen_policy_sha256') != manifest['policy_sha256']):
            raise ValueError('invalid screened candidate')
        candidates[row['id']] = row
    by_project, skipped, redactions, seen, source_hashes = defaultdict(list), Counter(), Counter(), set(), []
    normalized = set(cfg.get('exclude_normalized_sha256', []))
    for source in plan['sources']:
        if not isinstance(source, dict) or source.get('split') not in ('train', 'validation', 'test'):
            raise ValueError('invalid declared source split')
        if source['split'] != 'train':
            continue
        path = Path(source['path'])
        raw = screen._read(path, cfg['max_source_bytes'], 'source_unavailable')
        source_hash = sha(raw)
        source_hashes.append((path, source_hash))
        for line in raw.splitlines():
            original = json.loads(line)
            row = candidates.get(original.get('id')) if isinstance(original, dict) else None
            if (row is None or row['id'] in excluded
                    or not cfg['min_followup_bytes'] <= len(row['text'].encode()) <= cfg['max_followup_bytes']):
                continue
            if row['id'] in seen:
                raise ValueError('duplicate source identity')
            seen.add(row['id'])
            if (row.get('candidate_file_sha256') != source_hash or row.get('source_record_sha256') != sha(line)
                    or row.get('project') != source.get('project') or row.get('project_group') != source.get('project_group')
                    or not isinstance(original.get('text'), str) or sha(original['text'].encode()) != row.get('source_text_sha256')):
                raise ValueError('candidate provenance does not match train source')
            origin = original.get('source')
            if not isinstance(origin, dict):
                raise ValueError('candidate source provenance is missing')
            try:
                if origin.get('kind') == 'codex-session-user-text':
                    request, prior, truncated = codex.prior_user_context(
                        {**origin, 'schema_version': 'lfg.unreviewed-codex-message.v2'}, cfg['max_session_bytes'],
                        cfg['max_line_bytes'], cfg['max_prior_messages'], cfg['max_prior_bytes'])
                elif origin.get('kind') == 'claude-session-human-text':
                    request, prior, truncated = claude.prior_user_context(
                        {**origin, 'schema_version': 'lfg.unreviewed-claude-message.v1'}, cfg['max_session_bytes'],
                        cfg['max_line_bytes'], cfg['max_prior_messages'], cfg['max_prior_bytes'], cfg['max_ancestry_records'])
                else:
                    skipped['unsupported_source'] += 1
                    continue
            except (OSError, ValueError, KeyError, TypeError, UnicodeError):
                skipped['source_verification_failed'] += 1
                continue
            if request != original['text']:
                raise ValueError('verified original request differs from candidate')
            if not prior:
                skipped['no_prior_user_context'] += 1
                continue
            assembled = cfg['prior_prefix'] + cfg['separator'].join(x['text'] for x in prior) + cfg['current_prefix'] + request
            try:
                text, counts = screen._screen(assembled, rules, policy['limits'])
            except screen.ScreenError:
                skipped['screening_failed'] += 1
                continue
            if not text.strip() or len(text.encode()) > cfg['max_request_bytes']:
                skipped['request_too_large'] += 1
                continue
            key = corpus_audit.text_key(text)
            if key in normalized:
                skipped['duplicate_normalized_request'] += 1
                continue
            normalized.add(key)
            item = {'id': row['id'], 'project': row['project'], 'project_group': row['project_group'],
                    'family': row['family'], 'text': text, 'screened_text_sha256': sha(text.encode()),
                    'source_text_sha256_claim': row['source_text_sha256'], 'candidate_file_sha256_claim': source_hash,
                    'context_source_prefix_sha256_claim': origin['source_prefix_sha256'],
                    'context_message_count': len(prior), 'context_truncated': truncated,
                    'review_status': 'unreviewed', 'outcome': 'unknown', 'split': 'train'}
            by_project[row['project']].append((sha((cfg['seed'] + '\0' + row['id']).encode()), item, counts))
    for entries in by_project.values():
        entries.sort(key=lambda item: item[0])
    selected = []
    for index in range(cfg['max_per_project']):
        for project in sorted(by_project):
            if index < len(by_project[project]) and len(selected) < cfg['max_selected']:
                _, item, counts = by_project[project][index]
                selected.append(item)
                redactions.update(counts)
    if not selected:
        raise ValueError('no eligible contextual requests')
    if (screen._read(config_path, 1048576, 'config_changed') != config_raw
            or Path(__file__).read_bytes() != code_raw
            or screen._read(plan_path, cfg['max_manifest_bytes'], 'plan_changed') != plan_raw
            or screen._read(dataset / 'manifest.json', cfg['max_manifest_bytes'], 'manifest_changed') != manifest_raw
            or screen._read(dataset / 'train.jsonl', cfg['max_train_bytes'], 'train_changed') != train_raw
            or screen.load_policy(policy_path, cfg['max_manifest_bytes'])[2] != policy_sha
            or any(sha(Path(module.__file__).read_bytes()) != helper_hashes[Path(module.__file__).name] for module in helpers)
            or any(sha(screen._read(path, cfg['max_source_bytes'], 'source_changed')) != digest for path, digest in source_hashes)):
        raise ValueError('selection inputs changed')
    records = b''.join(encoded(row) for row in selected)
    report = {'schema_version': 'lfg.teacher-request-selection-result.v1', 'selection_config_sha256': sha(config_raw),
              'screened_manifest_sha256': sha(manifest_raw), 'screened_train_sha256': sha(train_raw),
              'source_plan_sha256': sha(plan_raw), 'screen_policy_sha256': policy_sha,
              'code_sha256': sha(code_raw), 'helper_sha256': helper_hashes, 'requests_sha256': sha(records),
              'requests': len(selected), 'projects': dict(sorted(Counter(row['project'] for row in selected).items())),
              'skipped': dict(sorted(skipped.items())), 'redaction_counts': dict(sorted(redactions.items())),
              'validation_opened': False, 'test_opened': False, 'redaction_complete': False,
              'human_reviewed': False, 'promotion_eligible': False}
    private_directory(output)
    output.mkdir(mode=0o700)
    for name, raw in (('requests.jsonl', records), ('manifest.json', encoded(report))):
        with (output / name).open('xb') as stream:
            stream.write(raw)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'dataset', 'policy', 'config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = select(args.plan, args.dataset, args.policy, args.config, args.output)
        print(json.dumps({k: result[k] for k in ('requests', 'projects', 'skipped', 'validation_opened', 'test_opened')}))
    except (OSError, ValueError, TypeError, KeyError, UnicodeError) as error:
        raise SystemExit(f'Contextual selection failed ({type(error).__name__})') from None


if __name__ == '__main__':
    main()
