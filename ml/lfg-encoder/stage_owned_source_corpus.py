#!/usr/bin/env python3
"""Screen explicitly selected tracked source files into a private train corpus."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess

import screen_private_corpus as screen
from owned_sft_checkpoint import private_directory


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stage(config_path: Path, policy_path: Path, output: Path) -> dict:
    config_raw = config_path.read_bytes()
    if len(config_raw) > 1048576:
        raise ValueError('source configuration is too large')
    config = json.loads(config_raw)
    if (not isinstance(config, dict) or config.get('schema_version') != 'lfg.owned-source-selection.v1'
            or config.get('sampling') not in ('byte_uniform', 'document_uniform', 'group_uniform')
            or type(config.get('min_document_bytes')) is not int or config['min_document_bytes'] < 1
            or type(config.get('max_total_bytes')) is not int or config['max_total_bytes'] < 1
            or type(config.get('max_files')) is not int or config['max_files'] < 1
            or not isinstance(config.get('projects'), list) or not config['projects']):
        raise ValueError('invalid source selection configuration')
    policy, rules, policy_sha = screen.load_policy(policy_path, 1048576)
    if output.exists():
        raise ValueError('output already exists')
    private_directory(output)
    output.mkdir(mode=0o700)
    sources, provenance, skipped, redactions = [], [], Counter(), Counter()
    total, seen, projects, snapshots = 0, set(), set(), []
    for project in config['projects']:
        if (not isinstance(project, dict) or any(not isinstance(project.get(k), str) or not project[k]
                                                 for k in ('id', 'group', 'root'))
                or project['id'] in projects or not Path(project['root']).is_absolute()
                or not isinstance(project.get('include_paths'), list) or not project['include_paths']
                or not isinstance(project.get('extensions'), list) or not project['extensions']
                or not isinstance(project.get('exclude_substrings', []), list)
                or any(not isinstance(x, str) or not x for k in ('include_paths', 'extensions', 'exclude_substrings')
                       for x in project.get(k, []))):
            raise ValueError('invalid project source selection')
        projects.add(project['id'])
        root = Path(project['root']).resolve(strict=True)
        for include in project['include_paths']:
            candidate = Path(include)
            if candidate.is_absolute() or '..' in candidate.parts:
                raise ValueError('include path must remain within its project')
        head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        listing = subprocess.check_output(['git', '-C', str(root), 'ls-files', '-z', '--', *project['include_paths']])
        for raw_name in listing.split(b'\0'):
            if not raw_name:
                continue
            name = raw_name.decode('utf-8')
            if not any(name.lower().endswith(ext.lower()) for ext in project['extensions']) or any(
                    item.casefold() in name.casefold() for item in project.get('exclude_substrings', [])):
                continue
            path = root / name
            if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root):
                skipped['unavailable_or_link'] += 1
                continue
            size = path.stat().st_size
            if size > min(policy['limits']['max_source_bytes'], policy['limits']['max_text_bytes']):
                skipped['oversized'] += 1
                continue
            raw = path.read_bytes()
            total += len(raw)
            if total > config['max_total_bytes']:
                raise ValueError('source corpus exceeds configured total byte limit')
            try:
                cleaned, counts = screen._screen(raw.decode('utf-8'), rules, policy['limits'])
            except (UnicodeError, screen.ScreenError):
                skipped['screening_failed'] += 1
                continue
            data = cleaned.encode('utf-8')
            if len(data) < config['min_document_bytes']:
                skipped['too_short'] += 1
                continue
            data_sha = sha(data)
            if data_sha in seen:
                skipped['duplicate'] += 1
                continue
            if len(sources) >= config['max_files']:
                raise ValueError('source corpus exceeds configured file count')
            seen.add(data_sha)
            name_out = f'document-{len(sources) + 1:06d}.utf8'
            descriptor = os.open(output / name_out, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, 'wb') as stream:
                stream.write(data)
            sources.append({'path': name_out, 'project': project['id'], 'project_group': project['group'],
                            'split': 'train', 'sha256': data_sha})
            provenance.append({'document': name_out, 'project': project['id'], 'source_path': name,
                               'git_head_claim': head, 'source_sha256': sha(raw), 'screened_sha256': data_sha,
                               'source_bytes': len(raw), 'screened_bytes': len(data)})
            snapshots.append((path, sha(raw)))
            redactions.update(counts)
    if not sources:
        raise ValueError('no source documents passed screening')
    if (config_path.read_bytes() != config_raw or screen.load_policy(policy_path, 1048576)[2] != policy_sha
            or any(not path.is_file() or path.is_symlink() or sha(path.read_bytes()) != digest
                   for path, digest in snapshots)):
        raise ValueError('source selection changed during staging')
    plan = {'schema_version': 'lfg.document-development-plan.v1', 'sampling': config['sampling'], 'sources': sources}
    (output / 'plan.json').write_text(json.dumps(plan, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    result = {'schema_version': 'lfg.owned-source-corpus.v1', 'config_sha256': sha(config_raw),
              'policy_sha256': policy_sha, 'documents': len(sources), 'source_bytes': total,
              'screened_bytes': sum(item['screened_bytes'] for item in provenance),
              'projects': dict(sorted(Counter(item['project'] for item in provenance).items())),
              'skipped': dict(sorted(skipped.items())), 'redaction_counts': dict(sorted(redactions.items())),
              'sources': provenance, 'redaction_complete': False, 'human_reviewed': False,
              'validation_opened': False, 'test_opened': False}
    (output / 'manifest.json').write_text(json.dumps(result, sort_keys=True, indent=2) + '\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'policy', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    result = stage(args.config, args.policy, args.output)
    print(json.dumps({key: result[key] for key in ('documents', 'source_bytes', 'screened_bytes', 'projects',
                                                    'skipped', 'redaction_counts', 'test_opened')}))


if __name__ == '__main__':
    main()
