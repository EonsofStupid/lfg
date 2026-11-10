#!/usr/bin/env python3
"""Train unqualified candidate-conditioned assessment heads."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
import platform
import subprocess
from pathlib import Path

import assessment_evaluate
import assessment_targets
import corpus_audit
import train
import training_inputs

GROUPS = ('complexity', 'models', 'effort', 'agent_count')


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def encoded(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)


def load_config(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    config = json.loads(raw)
    limits = {'max_tokens', 'max_text_bytes', 'batch_size', 'cpu_threads', 'max_dimensions',
              'max_records_per_split', 'max_pairs_per_split', 'max_unique_texts', 'max_fit_bytes', 'max_pack_bytes'}
    fields = limits | {'schema_version', 'revision', 'device', 'ridge', 'include_subscription_ids', 'requirements'}
    if (not isinstance(config, dict) or set(config) != fields
            or config['schema_version'] != 'lfg.assessment-training-config.v1'
            or not isinstance(config['revision'], str) or not config['revision'].strip()
            or config['device'] not in ('cpu', 'cuda')
            or type(config['ridge']) not in (int, float) or not math.isfinite(config['ridge']) or config['ridge'] <= 0
            or type(config['include_subscription_ids']) is not bool
            or any(type(config[key]) is not int or config[key] < 1 for key in limits)
            or not isinstance(config['requirements'], dict) or set(config['requirements']) != set(GROUPS)):
        raise ValueError('invalid assessment training configuration')
    for group, requirements in config['requirements'].items():
        keys = ({'train_known', 'validation_known'} if group == 'agent_count' else
                {'train_records', 'train_positive', 'train_negative',
                 'validation_records', 'validation_positive', 'validation_negative'})
        if (not isinstance(requirements, dict) or set(requirements) != keys
                or any(type(value) is not int or value < 1 for value in requirements.values())):
            raise ValueError('invalid assessment support requirements')
    if path.read_bytes() != raw:
        raise ValueError('assessment training configuration changed')
    return config, digest(raw)


def model_descriptor(model: dict, include_subscription_ids: bool) -> dict:
    result = {key: model[key] for key in ('provider', 'model', 'label')}
    if include_subscription_ids:
        result['subscription_id'] = model['subscription_id']
    return result


def profile_context(profile: dict, include_subscription_ids: bool) -> str:
    available = [model for model in profile['models'] if model['availability'] == 'available']
    descriptions = [{**model_descriptor(model, include_subscription_ids), 'effort_levels': sorted(model['effort_levels'])}
                    for model in available]
    return encoded({'complexities': sorted(option['label'] for option in profile['complexities']),
                    'models': sorted(descriptions, key=encoded), 'agent_range': profile['agent_range']})


def project_examples(rows: list[dict], config: dict) -> tuple[dict, dict]:
    """Create in-memory candidate pairs; unknown judgments produce no labels."""
    if len(rows) > config['max_records_per_split']:
        raise ValueError('assessment split exceeds record limit')
    output = {group: [] for group in GROUPS}
    support = {group: Counter() for group in GROUPS}
    pair_count = 0
    def add(group: str, item: tuple) -> None:
        nonlocal pair_count
        pair_count += 1
        if pair_count > config['max_pairs_per_split']:
            raise ValueError('assessment split exceeds pair limit')
        output[group].append(item)
    include = config['include_subscription_ids']
    for row in rows:
        targets = row.get('assessment_targets')
        if targets is None:
            continue
        profile, values = targets['project_profile'], targets['values']
        assessment_targets.validate_stored(targets, row['source']['project'])
        query, context = row['text'], profile_context(profile, include)
        complexities = [(option['id'], encoded({'kind': 'complexity', 'label': option['label']}))
                        for option in profile['complexities']]
        models = [(model['id'], encoded({'kind': 'model', **model_descriptor(model, include)}), model)
                  for model in profile['models'] if model['availability'] == 'available']
        if (len({description for _, description in complexities}) != len(complexities)
                or len({description for _, description, _ in models}) != len(models)):
            raise ValueError('project profile has indistinguishable candidate descriptions')
        if values['complexity'] is not None:
            support['complexity']['known'] += 1
            support['complexity']['records'] += 1
            for identifier, candidate in complexities:
                label = int(identifier == values['complexity'])
                add('complexity', (query, context, candidate, label))
                support['complexity']['positive' if label else 'negative'] += 1
        if values['models'] is not None:
            support['models']['known'] += 1
            support['models']['records'] += 1
            selected = {item['id']: item['effort'] for item in values['models']}
            if any(effort is not None for effort in selected.values()):
                support['effort']['records'] += 1
            for identifier, candidate, model in models:
                label = int(identifier in selected)
                add('models', (query, context, candidate, label))
                support['models']['positive' if label else 'negative'] += 1
                if identifier in selected and selected[identifier] is not None:
                    support['effort']['known'] += 1
                    for effort in model['effort_levels']:
                        effort_candidate = encoded({'kind': 'effort', **model_descriptor(model, include), 'effort': effort})
                        positive = int(effort == selected[identifier])
                        add('effort', (query, context, effort_candidate, positive))
                        support['effort']['positive' if positive else 'negative'] += 1
        if values['agent_count'] is not None:
            support['agent_count']['known'] += 1
            add('agent_count', (query, context, None, values['agent_count']))
    return output, {key: dict(value) for key, value in support.items()}


def checked_inputs(plan_path: Path, config: dict) -> tuple[dict, dict, dict]:
    audit, validation = assessment_evaluate.validation_records(plan_path)
    plan_raw = plan_path.read_bytes()
    if digest(plan_raw) != audit['plan_sha256']:
        raise ValueError('assessment plan changed')
    plan = json.loads(plan_raw)
    training = []
    for source in plan['sources']:
        if source['split'] == 'train':
            rows, _ = train.load_dataset(corpus_audit.resolve(plan_path.parent, source['path']))
            training.extend(rows)
    examples = {}
    support = {}
    for split, rows in (('train', training), ('validation', [item['row'] for item in validation])):
        examples[split], support[split] = project_examples(rows, config)
    for group, required in config['requirements'].items():
        for key, minimum in required.items():
            split, field = key.split('_', 1)
            if support[split][group].get(field, 0) < minimum:
                raise ValueError(f'assessment {group} {key} support is below the configured minimum')
    if assessment_evaluate.validation_records(plan_path)[0] != audit:
        raise ValueError('assessment inputs changed during support check')
    return audit, examples, support


def exploratory_inputs(plan_path: Path, config: dict) -> tuple[dict, dict, dict]:
    """Load explicit target values without claiming review or qualification."""
    raw = plan_path.read_bytes()
    plan = json.loads(raw)
    sources = plan.get('sources') if isinstance(plan, dict) else None
    if (not isinstance(plan, dict) or plan.get('schema_version') != corpus_audit.DEVELOPMENT_PLAN_SCHEMA
            or not isinstance(sources, list) or not sources
            or any(not isinstance(source, dict) or source.get('split') not in ('train', 'validation') for source in sources)
            or not any(source['split'] == 'train' for source in sources)):
        raise ValueError('exploratory training requires a development plan with training sources only or train and validation')
    audit, _ = corpus_audit.audit(plan_path, development_only=True)
    if digest(raw) != audit['plan_sha256']:
        raise ValueError('assessment plan changed')
    rows = {'train': [], 'validation': []}
    for source, evidence in zip(sources, audit['sources'], strict=True):
        loaded, source_sha = train.load_dataset(corpus_audit.resolve(plan_path.parent, source['path']),
                                                allow_unlabeled_review=True)
        if source_sha != evidence['sha256']:
            raise ValueError('assessment source changed')
        rows[source['split']].extend(loaded)
    examples, support = {}, {}
    for split in ('train', 'validation'):
        examples[split], support[split] = project_examples(rows[split], config)
    if not any(examples['train'][group] for group in GROUPS):
        raise ValueError('exploratory training requires at least one explicit assessment target')
    if corpus_audit.audit(plan_path, development_only=True)[0] != audit:
        raise ValueError('assessment inputs changed during exploratory preparation')
    return audit, examples, support


def support_failures(support: dict, config: dict) -> list[dict]:
    failures = []
    for group, required in config['requirements'].items():
        for key, minimum in required.items():
            split, field = key.split('_', 1)
            actual = support[split][group].get(field, 0)
            if actual < minimum:
                failures.append({'target': group, 'requirement': key, 'actual': actual, 'minimum': minimum})
    return failures


def require_private_file(path: Path) -> None:
    """Within a Git worktree the exact output file must be ignored."""
    path = path.resolve()
    ancestor = path.parent
    while not ancestor.exists():
        ancestor = ancestor.parent
    try:
        result = subprocess.run(['git', '-C', str(ancestor), 'rev-parse', '--show-toplevel'],
                                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False, text=True)
        if result.returncode != 0:
            if any((directory / '.git').exists() for directory in (ancestor, *ancestor.parents)):
                raise ValueError('cannot verify output privacy')
            return
        root = Path(result.stdout.rstrip('\r\n')).resolve()
        if not path.is_relative_to(root):
            raise ValueError('cannot verify output privacy')
        relative = path.relative_to(root).as_posix()
        tracked = subprocess.run(['git', '-C', str(root), 'ls-files', '--error-unmatch', '--', relative],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if tracked.returncode == 0:
            raise ValueError('assessment output is already tracked by Git')
        if tracked.returncode != 1:
            raise ValueError('cannot verify output privacy')
        ignored = subprocess.run(['git', '-C', str(root), 'check-ignore', '-q', '--no-index', '--',
                                  relative],
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        if ignored.returncode != 0:
            raise ValueError('assessment output in Git must be ignored')
    except OSError:
        raise ValueError('cannot verify output privacy') from None


class Encoder:
    def __init__(self, model: Path, config: dict):
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.config = torch, config
        if config['device'] == 'cuda' and not torch.cuda.is_available():
            raise ValueError('requested CUDA device is unavailable')
        torch.set_num_threads(config['cpu_threads'])
        torch.use_deterministic_algorithms(True)
        if config['device'] == 'cuda':
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.device = torch.device(config['device'])
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(model, local_files_only=True, trust_remote_code=False,
                                               use_safetensors=True).to(self.device).eval()
        if next(self.model.parameters()).device.type != config['device']:
            raise ValueError('encoder device differs from requested device')
        capacity = getattr(self.model.config, 'max_position_embeddings', None)
        if isinstance(capacity, int) and capacity > 0 and config['max_tokens'] > capacity:
            raise ValueError('configured token limit exceeds encoder capacity')
        dimension = getattr(self.model.config, 'hidden_size', None)
        if isinstance(dimension, int) and dimension > config['max_dimensions']:
            raise ValueError('encoder dimension exceeds configured limit')
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.versions = {'torch': str(torch.__version__), 'transformers': transformers.__version__,
                         'python': platform.python_version(), 'platform': platform.platform(),
                         'cuda_runtime': torch.version.cuda,
                         'device_name': torch.cuda.get_device_name(self.device) if config['device'] == 'cuda' else None,
                         'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                         'deterministic_algorithms': torch.are_deterministic_algorithms_enabled(),
                         'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG')}

    def encode(self, texts: list[str]) -> dict[str, list[float]]:
        if not texts or len(texts) > self.config['max_unique_texts']:
            raise ValueError('invalid number of unique encoder inputs')
        for text in texts:
            if not isinstance(text, str) or not text.strip() or len(text.encode('utf-8')) > self.config['max_text_bytes']:
                raise ValueError('empty or oversized encoder input')
            tokens = self.tokenizer(text, truncation=False, add_special_tokens=True)['input_ids']
            if not tokens or len(tokens) > self.config['max_tokens']:
                raise ValueError('encoder input exceeds token limit; truncation is disabled')
        vectors = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), self.config['batch_size']):
                batch = self.tokenizer(texts[start:start + self.config['batch_size']], padding=True,
                                       truncation=False, return_tensors='pt')
                batch = {key: value.to(self.device) for key, value in batch.items()}
                hidden = self.model(**batch).last_hidden_state
                mask = batch['attention_mask'].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                norm = self.torch.linalg.vector_norm(pooled, dim=1, keepdim=True)
                if not bool(self.torch.isfinite(pooled).all().item()) or bool((norm <= 0).any().item()):
                    raise ValueError('encoder produced non-finite or zero vectors')
                vectors.extend((pooled / norm).cpu().tolist())
        if (not vectors or len(vectors[0]) > self.config['max_dimensions'] or not vectors[0]
                or any(len(vector) != len(vectors[0]) or any(not math.isfinite(value) for value in vector)
                       for vector in vectors)):
            raise ValueError('invalid encoder dimensions or vectors')
        return dict(zip(texts, vectors, strict=True))


def fit(examples: dict, vectors: dict, config: dict, allow_partial: bool = False) -> dict:
    import torch
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        raise ValueError('requested CUDA device is unavailable')
    device = torch.device(config['device'])
    if not vectors:
        raise ValueError('assessment training requires vectors')
    texts = sorted(vectors)
    dimensions = len(vectors[texts[0]])
    if (not dimensions or dimensions > config['max_dimensions']
            or any(len(vector) != dimensions or any(type(value) not in (int, float) or not math.isfinite(value)
                                                    for value in vector) for vector in vectors.values())):
        raise ValueError('invalid assessment training vectors')
    width = 4 * dimensions + 1
    groups = [group for group in GROUPS if examples[group]] if allow_partial else list(GROUPS)
    if not groups:
        raise ValueError('assessment training requires at least one target')
    for group in groups:
        n = len(examples[group])
        if not n:
            raise ValueError('assessment target has no training examples')
        # Bound a conservative tensor estimate; allocator, model and solver workspaces are additional.
        estimate = 8 * (len(texts) * dimensions + 3 * n * width + min(n, width) ** 2)
        if estimate > config['max_fit_bytes']:
            raise ValueError('assessment fit exceeds configured tensor estimate')
    indices = {text: index for index, text in enumerate(texts)}
    embeddings = torch.tensor([vectors[text] for text in texts], dtype=torch.float64, device=device)
    weights = {}
    for group in groups:
        rows = examples[group]
        query_indices = torch.tensor([indices[row[0]] for row in rows], dtype=torch.long, device=device)
        context_indices = torch.tensor([indices[row[1]] for row in rows], dtype=torch.long, device=device)
        q, p = embeddings[query_indices], embeddings[context_indices]
        if group == 'agent_count':
            c = torch.zeros_like(q)
        else:
            candidate_indices = torch.tensor([indices[row[2]] for row in rows], dtype=torch.long, device=device)
            c = embeddings[candidate_indices]
        matrix = torch.cat((q, p, c, q * c, torch.ones((len(rows), 1), dtype=torch.float64, device=device)), dim=1)
        target = torch.tensor([float(row[3]) for row in rows], dtype=torch.float64, device=device)
        # Solve in the smaller space; both forms minimize squared error plus ridge * ||weights||².
        if matrix.shape[0] <= matrix.shape[1]:
            gram = matrix @ matrix.T + config['ridge'] * torch.eye(matrix.shape[0], dtype=matrix.dtype, device=device)
            result = matrix.T @ torch.linalg.solve(gram, target)
        else:
            gram = matrix.T @ matrix + config['ridge'] * torch.eye(matrix.shape[1], dtype=matrix.dtype, device=device)
            result = torch.linalg.solve(gram, matrix.T @ target)
        if result.device.type != config['device'] or not bool(torch.isfinite(result).all().item()):
            raise ValueError('assessment weights are non-finite or on an unexpected device')
        weights[group] = result.cpu().tolist()
    return {'dimension': dimensions, 'weights': weights}


def build(args: argparse.Namespace) -> dict:
    code = [Path(__file__), Path(assessment_evaluate.__file__), Path(assessment_targets.__file__),
            Path(corpus_audit.__file__), Path(train.__file__), Path(training_inputs.__file__),
            Path(training_inputs.lock_wheelhouse.__file__), Path(corpus_audit.evaluate.__file__)]
    code_hashes = {path.name: digest(path.read_bytes()) for path in code}
    config, config_sha = load_config(args.config)
    exploratory = getattr(args, 'exploratory', False)
    inputs = exploratory_inputs if exploratory else checked_inputs
    audit, examples, support = inputs(args.plan, config)
    if (args.output.exists() or args.output.resolve().is_relative_to(args.model.resolve())
            or args.output.resolve().is_relative_to(args.plan.parent.resolve())):
        raise ValueError('output must be new and outside the model snapshot and corpus plan directory')
    require_private_file(args.output)
    model = training_inputs.verify_model(args.model, args.model_manifest, args.revision)
    environment = training_inputs.verify_environment(args.environment_lock, args.wheelhouse, args.allowed_extra,
                                                     verify_payload=True)
    train_examples = examples['train']
    texts = sorted({text for rows in train_examples.values() for query, context, candidate, _ in rows
                    for text in (query, context, candidate) if text is not None})
    encoder = Encoder(args.model, config)
    vectors = encoder.encode(texts)
    learned = fit(train_examples, vectors, config, allow_partial=exploratory)
    if (inputs(args.plan, config)[0] != audit or load_config(args.config)[1] != config_sha
            or os.environ.get('CUBLAS_WORKSPACE_CONFIG') != encoder.versions['cublas_workspace_config']
            or os.environ.get('CUDA_VISIBLE_DEVICES') != encoder.versions['cuda_visible_devices']
            or training_inputs.verify_model(args.model, args.model_manifest, args.revision) != model
            or training_inputs.verify_environment(args.environment_lock, args.wheelhouse, args.allowed_extra,
                                                   verify_payload=True) != environment
            or any(digest(path.read_bytes()) != code_hashes[path.name] for path in code)):
        raise ValueError('assessment training inputs changed during build')
    trained_targets = list(learned['weights'])
    pack = {'schema_version': 'lfg.assessment-heads-exploratory.v1' if exploratory else 'lfg.assessment-heads.v1',
            'promotion_eligible': False,
            'feature_schema': 'lfg.assessment-candidate-features.v1', 'trained_targets': trained_targets,
            'audit': audit, 'support': support, 'config': config, 'config_sha256': config_sha,
            'model': model, 'environment': environment, 'encoder_versions': encoder.versions,
            'code_sha256': code_hashes, **learned}
    if exploratory:
        pack.update(training_mode='exploratory', review_required=False,
                    support_failures=support_failures(support, config),
                    validation_used_for_training=False)
    raw = (encoded(pack) + '\n').encode('utf-8')
    if len(raw) > config['max_pack_bytes']:
        raise ValueError('assessment weight pack exceeds configured limit')
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, 'wb') as stream:
        stream.write(raw)
    return {'schema_version': 'lfg.assessment-training-result.v1', 'pack_sha256': digest(raw),
            'support': support, 'trained_targets': trained_targets, 'promotion_eligible': False,
            'training_mode': 'exploratory' if exploratory else 'reviewed'}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('plan', 'config', 'model', 'model-manifest', 'environment-lock', 'wheelhouse', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--allowed-extra', action='append', default=[])
    parser.add_argument('--exploratory', action='store_true',
                        help='fit explicit available targets without review or support thresholds; never promotion-eligible')
    args = parser.parse_args()
    try:
        print(json.dumps(build(args), sort_keys=True, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, UnicodeError) as error:
        raise SystemExit(f'assessment training rejected: {error}') from error


if __name__ == '__main__':
    main()
