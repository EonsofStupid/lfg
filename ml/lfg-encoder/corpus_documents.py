"""Load explicitly authorized, locally screened development documents."""
from __future__ import annotations

from bisect import bisect_right
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

import screen_private_corpus as screening


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def load(plan_path: Path, policy_path: Path, max_data_bytes: int, context: int) -> dict:
    policy, rules, policy_sha = screening.load_policy(policy_path, max_data_bytes)
    limits = policy['limits']
    raw_plan = screening._read(plan_path, limits['max_plan_bytes'], 'corpus_plan_unavailable')
    plan = screening._json(raw_plan, 'invalid_corpus_plan')
    sources = plan.get('sources') if isinstance(plan, dict) else None
    if (not isinstance(plan, dict) or set(plan) not in ({'schema_version', 'sources'}, {'schema_version', 'sources', 'sampling'})
            or plan.get('schema_version') != 'lfg.document-development-plan.v1'
            or plan.get('sampling', 'byte_uniform') not in ('byte_uniform', 'document_uniform', 'group_uniform')
            or not isinstance(sources, list) or not sources
            or any(not isinstance(source, dict) or set(source) != {'path', 'project', 'project_group', 'split', 'sha256'}
                   or source.get('split') not in ('train', 'validation')
                   or any(not isinstance(source.get(key), str) or not source[key].strip()
                          for key in ('path', 'project', 'project_group'))
                   or not isinstance(source.get('sha256'), str)
                   or not re.fullmatch(r'[0-9a-f]{64}', source['sha256']) for source in sources)):
        raise ValueError('invalid development plan or held-out split')
    projects, groups, hashes, paths = {}, {}, {}, {}
    for item in sources:
        project, group, split, sha = item['project'], item['project_group'], item['split'], item['sha256']
        path = (plan_path.parent / item['path']).resolve()
        if ((project in projects and projects[project] != (group, split))
                or (group in groups and groups[group] != split)
                or (sha in hashes and hashes[sha] != split)
                or (path in paths and paths[path] != split)):
            raise ValueError('project, group, content or path crosses split')
        projects[project], groups[group], hashes[sha], paths[path] = (group, split), split, split, split
    selected = [item for item in sources if item['split'] == 'train']
    if not selected or len(selected) > limits['max_records']:
        raise ValueError('training source count is invalid')
    documents, entries, snapshots, total_bytes, total_output, redaction_counts = [], [], [], 0, 0, Counter()
    screened_hashes = set()
    for item in selected:
        path = (plan_path.parent / item['path']).resolve()
        raw = screening._read(path, limits['max_source_bytes'], 'training_source_unavailable')
        total_bytes += len(raw)
        if total_bytes > min(max_data_bytes, limits['max_total_source_bytes']):
            raise ValueError('training corpus exceeds configured byte limit')
        original_sha = digest(raw)
        if original_sha != item['sha256']:
            raise ValueError('training source identity differs from plan')
        try:
            text = raw.decode('utf-8')
        except UnicodeError:
            raise ValueError('training source is not UTF-8') from None
        if len(raw) > limits['max_text_bytes']:
            raise ValueError('training source exceeds text limit')
        screened, counts = screening._screen(text, rules, limits)
        data = screened.encode('utf-8')
        total_output += len(data)
        if total_output > min(max_data_bytes, limits['max_total_output_bytes']):
            raise ValueError('screened corpus exceeds configured byte limit')
        if len(data) <= context:
            raise ValueError('screened training document is too short for the context')
        if digest(data) in screened_hashes:
            raise ValueError('duplicate screened training document')
        screened_hashes.add(digest(data))
        redaction_counts.update(counts)
        documents.append(data)
        entries.append({'project': item['project'], 'project_group': item['project_group'],
                        'original_sha256': original_sha, 'screened_sha256': digest(data), 'screened_bytes': len(data)})
        snapshots.append((path, original_sha))
    identity = {'schema_version': 'lfg.screened-document-corpus.v1', 'plan_sha256': digest(raw_plan),
                'policy_sha256': policy_sha, 'sources': entries, 'sampling': plan.get('sampling', 'byte_uniform'),
                'redaction_counts': dict(sorted(redaction_counts.items())),
                'declared_validation_sources': [{'project': item['project'], 'project_group': item['project_group'],
                                                 'sha256': item['sha256']} for item in sources if item['split'] == 'validation'],
                'redaction_complete': False, 'human_reviewed': False, 'validation_sources_opened': False}
    canonical = json.dumps(identity, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    return {'documents': documents, 'identity': identity, 'corpus_sha256': digest(canonical),
            'snapshots': snapshots, 'plan_path': plan_path, 'policy_path': policy_path,
            'plan_sha256': digest(raw_plan), 'policy_sha256': policy_sha,
            'max_plan_bytes': limits['max_plan_bytes'], 'max_source_bytes': limits['max_source_bytes'],
            'policy_read_limit': max_data_bytes}


def verify(corpus: dict) -> None:
    if (digest(screening._read(corpus['plan_path'], corpus['max_plan_bytes'], 'plan_changed')) != corpus['plan_sha256']
            or screening.load_policy(corpus['policy_path'], corpus['policy_read_limit'])[2] != corpus['policy_sha256']
            or any(digest(screening._read(path, corpus['max_source_bytes'], 'source_changed')) != sha
                   for path, sha in corpus['snapshots'])):
        raise ValueError('corpus inputs changed')


def sampler(corpus: dict, context: int) -> dict:
    documents = corpus['documents']
    identity = corpus.get('identity') or (corpus.get('corpus') or {}).get('identity')
    sampling = identity['sampling'] if identity is not None else 'byte_uniform'
    groups = {}
    if sampling == 'group_uniform':
        for index, entry in enumerate(identity['sources']):
            groups.setdefault(entry['project_group'], []).append(index)
    def table(indices):
        cumulative, total = [], 0
        for index in indices:
            total += len(documents[index]) - context
            cumulative.append(total)
        return indices, cumulative, total
    return {'sampling': sampling, 'all': table(list(range(len(documents)))),
            'groups': [table(groups[name]) for name in sorted(groups)]}


def batch(corpus: dict, rng, batch_size: int, context: int, prepared: dict | None = None) -> tuple[list[list[int]], list[list[int]]]:
    documents = corpus['documents']
    prepared = prepared if prepared is not None else sampler(corpus, context)
    def weighted_index(table):
        indices, cumulative, total = table
        sample = rng.randrange(total)
        offset = bisect_right(cumulative, sample)
        return indices[offset], sample - (cumulative[offset - 1] if offset else 0)
    inputs, targets = [], []
    for _ in range(batch_size):
        if prepared['sampling'] == 'document_uniform':
            index = rng.randrange(len(documents))
            start = rng.randrange(len(documents[index]) - context)
        elif prepared['sampling'] == 'group_uniform':
            index, start = weighted_index(prepared['groups'][rng.randrange(len(prepared['groups']))])
        else:
            index, start = weighted_index(prepared['all'])
        data = documents[index]
        inputs.append(list(data[start:start + context]))
        targets.append(list(data[start + 1:start + context + 1]))
    return inputs, targets
