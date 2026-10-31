#!/usr/bin/env python3
"""Build an unpromoted per-node exemplar pack from explicitly reviewed data."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import node_review
import training_inputs


def read_config(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    value = json.loads(raw)
    fields = {'schemaVersion', 'revision', 'pooling', 'maxTokens', 'maxTextBytes', 'batchSize',
              'cpuThreads', 'maxExemplars', 'maxValidationItems', 'maxDimensions', 'maxPackBytes',
              'maxPredictionBytes', 'minimumSimilarity', 'minimumMargin'}
    if (not isinstance(value, dict) or set(value) != fields or value['schemaVersion'] != 'lfg.node-exemplar-config.v1'
            or not node_review.nonempty(value['revision']) or value['pooling'] != 'attention-mask-mean-l2'):
        raise ValueError('invalid exemplar configuration')
    for key in ('maxTokens', 'maxTextBytes', 'batchSize', 'cpuThreads', 'maxExemplars', 'maxValidationItems',
                'maxDimensions', 'maxPackBytes', 'maxPredictionBytes'):
        if type(value[key]) is not int or value[key] < 1:
            raise ValueError(f'invalid {key}')
    for key, low, high in (('minimumSimilarity', -1, 1), ('minimumMargin', 0, 2)):
        if type(value[key]) not in (int, float) or not math.isfinite(value[key]) or not low <= value[key] <= high:
            raise ValueError(f'invalid {key}')
    return value, node_review.digest(raw)


def encoder_config_sha(config: dict) -> str:
    keys = ('pooling', 'maxTokens', 'maxTextBytes', 'batchSize', 'cpuThreads')
    return node_review.digest(node_review.encode({key: config[key] for key in keys}))


class Encoder:
    def __init__(self, model: Path, config: dict):
        import torch
        import transformers
        from transformers import AutoModel, AutoTokenizer
        self.torch, self.config = torch, config
        torch.set_num_threads(config['cpuThreads'])
        self.tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True, trust_remote_code=False)
        self.model = AutoModel.from_pretrained(model, local_files_only=True, trust_remote_code=False,
                                               use_safetensors=True).to('cpu').eval()
        capacity = getattr(self.model.config, 'max_position_embeddings', None)
        if isinstance(capacity, int) and capacity > 0 and config['maxTokens'] > capacity:
            raise ValueError('configured token limit exceeds encoder capacity')
        dimension = getattr(self.model.config, 'hidden_size', None)
        if isinstance(dimension, int) and dimension > config['maxDimensions']:
            raise ValueError('encoder dimension exceeds configured limit')
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.versions = {'torch': torch.__version__, 'transformers': transformers.__version__}

    def encode(self, texts: list[str]) -> tuple[list[list[float]], list[int]]:
        if not texts or any(not isinstance(text, str) or not text.strip()
                            or len(text.encode('utf-8')) > self.config['maxTextBytes'] for text in texts):
            raise ValueError('empty or oversized encoder input')
        lengths = []
        for text in texts:
            encoded = self.tokenizer(text, truncation=False, add_special_tokens=True)
            size = len(encoded['input_ids'])
            if size < 1 or size > self.config['maxTokens']:
                raise ValueError('input exceeds configured token limit; truncation is disabled')
            lengths.append(size)
        vectors = []
        with self.torch.inference_mode():
            for start in range(0, len(texts), self.config['batchSize']):
                batch = self.tokenizer(texts[start:start + self.config['batchSize']], padding=True,
                                       truncation=False, return_tensors='pt')
                hidden = self.model(**batch).last_hidden_state
                mask = batch['attention_mask'].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                norms = self.torch.linalg.vector_norm(pooled, dim=1, keepdim=True)
                if not bool(self.torch.isfinite(pooled).all().item()) or bool((norms <= 0).any().item()):
                    raise ValueError('encoder produced non-finite or zero vector')
                output = pooled / norms
                if not bool(self.torch.isfinite(output).all().item()):
                    raise ValueError('encoder produced non-finite normalized vector')
                vectors.extend(output.tolist())
        dimension = len(vectors[0])
        if not dimension or dimension > self.config['maxDimensions'] or any(len(vector) != dimension for vector in vectors):
            raise ValueError('inconsistent embedding dimensions')
        return vectors, lengths


def choose(vector: list[float], exemplars: list[dict], options: list[str], config: dict) -> str | None:
    if len(options) < 2 or len(set(options)) != len(options) or not vector:
        raise ValueError('at least two distinct options and an embedding are required')
    def check_vector(values: object) -> None:
        if (not isinstance(values, list) or not values or any(type(x) not in (int, float) or not math.isfinite(x) for x in values)
                or abs(sum(x * x for x in values) - 1) > 0.0001):
            raise ValueError('embedding must be a finite unit vector')
    check_vector(vector)
    scores = {}
    for exemplar in exemplars:
        other = exemplar['embedding']
        check_vector(other)
        if exemplar['choice'] not in options or len(other) != len(vector):
            raise ValueError('invalid exemplar vector')
        similarity = max(-1.0, min(1.0, sum(a * b for a, b in zip(vector, other, strict=True))))
        scores[exemplar['choice']] = max(scores.get(exemplar['choice'], -math.inf), similarity)
    if set(scores) != set(options):
        raise ValueError('every option requires an exemplar')
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if ranked[0][1] < config['minimumSimilarity'] or ranked[0][1] - ranked[1][1] <= config['minimumMargin']:
        return None
    return ranked[0][0]


def model_proof(args: argparse.Namespace) -> dict:
    return training_inputs.verify_model(args.model, args.model_manifest, args.revision)


def input_audit(args: argparse.Namespace) -> tuple[dict, list[dict]]:
    report, records = node_review.audit(args.input, args.snapshot, args.policy, args.node)
    if not report['admissionEligible']:
        raise ValueError('node admission gates are not satisfied')
    return report, records


def environment(args: argparse.Namespace) -> dict:
    return training_inputs.verify_environment(args.environment_lock, args.wheelhouse, args.allowed_extra,
                                              verify_payload=True)


def output_path(args: argparse.Namespace) -> None:
    if args.output.exists() or args.output.resolve().is_relative_to(args.model.resolve()):
        raise ValueError('output must be a new file outside the model snapshot')


def bounded_read(path: Path, limit: int) -> bytes:
    with path.open('rb') as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError('exemplar pack exceeds configured byte limit')
    return raw


def build(args: argparse.Namespace) -> dict:
    tool_sha = node_review.digest(Path(__file__).read_bytes())
    verifier_sha = node_review.digest(Path(training_inputs.__file__).read_bytes())
    config, config_sha = read_config(args.config)
    audit, records = input_audit(args)
    training = sorted((r for r in records if r['review']['split'] == 'train'), key=lambda r: r['review']['itemSha256'])
    if not training or len(training) > config['maxExemplars']:
        raise ValueError('invalid number of training exemplars')
    output_path(args)
    model = model_proof(args)
    env = environment(args)
    encoder = Encoder(args.model, config)
    vectors, _ = encoder.encode([record['item']['text'] for record in training])
    options = sorted(json.loads(args.snapshot.read_bytes())['nodes'][args.node]['options'])
    entries = [{'itemSha256': row['review']['itemSha256'], 'choice': row['review']['choice'], 'embedding': vector}
               for row, vector in zip(training, vectors, strict=True)]
    choose(vectors[0], entries, options, config)
    if (input_audit(args)[0] != audit or model_proof(args) != model or environment(args) != env
            or read_config(args.config)[1] != config_sha or node_review.digest(Path(__file__).read_bytes()) != tool_sha
            or node_review.digest(Path(training_inputs.__file__).read_bytes()) != verifier_sha):
        raise ValueError('admission, model, environment or configuration changed during build')
    pack = {'schemaVersion': 'lfg.node-exemplar-pack.v1', 'promotionEligible': False, 'node': args.node,
            'snapshotSha256': audit['snapshotSha256'], 'admission': audit, 'model': model, 'environment': env,
            'encoderVersions': encoder.versions, 'buildConfigSha256': config_sha,
            'encoderConfigSha256': encoder_config_sha(config), 'options': options,
            'dimension': len(vectors[0]), 'exemplars': entries,
            'builderSha256': tool_sha, 'inputVerifierSha256': verifier_sha}
    raw = node_review.encode(pack) + b'\n'
    if len(raw) > config['maxPackBytes']:
        raise ValueError('exemplar pack exceeds configured byte limit')
    node_review.create_new(args.output, raw)
    return {'schemaVersion': 'lfg.node-exemplar-build.v1', 'packSha256': node_review.digest(raw),
            'exemplars': len(entries), 'dimension': pack['dimension'], 'promotionEligible': False}


def predict(args: argparse.Namespace) -> dict:
    tool_sha = node_review.digest(Path(__file__).read_bytes())
    verifier_sha = node_review.digest(Path(training_inputs.__file__).read_bytes())
    config, config_sha = read_config(args.config)
    audit, records = input_audit(args)
    validation = sorted((r for r in records if r['review']['split'] == 'validation'), key=lambda r: r['review']['itemSha256'])
    if not validation or len(validation) > config['maxValidationItems']:
        raise ValueError('invalid number of validation examples')
    raw_pack = bounded_read(args.pack, config['maxPackBytes'])
    if node_review.digest(raw_pack) != args.pack_sha256:
        raise ValueError('exemplar pack digest mismatch')
    pack = json.loads(raw_pack)
    if not isinstance(pack, dict):
        raise ValueError('invalid exemplar pack')
    expected = sorted((r['review']['itemSha256'], r['review']['choice']) for r in records if r['review']['split'] == 'train')
    entries = pack.get('exemplars')
    options = sorted(json.loads(args.snapshot.read_bytes())['nodes'][args.node]['options'])
    if (pack.get('schemaVersion') != 'lfg.node-exemplar-pack.v1' or pack.get('promotionEligible') is not False
            or pack.get('node') != args.node or pack.get('snapshotSha256') != audit['snapshotSha256']
            or pack.get('admission') != audit or pack.get('encoderConfigSha256') != encoder_config_sha(config) or pack.get('options') != options
            or pack.get('builderSha256') != tool_sha or pack.get('inputVerifierSha256') != verifier_sha
            or not isinstance(entries, list) or len(entries) > config['maxExemplars']
            or sorted((entry['itemSha256'], entry['choice']) for entry in entries) != expected):
        raise ValueError('exemplar pack does not match current reviewed inputs')
    output_path(args)
    model = model_proof(args)
    env = environment(args)
    if pack.get('model') != model or pack.get('environment') != env:
        raise ValueError('exemplar pack model or environment differs')
    encoder = Encoder(args.model, config)
    if pack.get('encoderVersions') != encoder.versions:
        raise ValueError('encoder versions differ from exemplar pack')
    vectors, _ = encoder.encode([record['item']['text'] for record in validation])
    if pack.get('dimension') != len(vectors[0]) or len(vectors[0]) > config['maxDimensions']:
        raise ValueError('embedding dimensions differ')
    choices = [choose(vector, entries, options, config) for vector in vectors]
    if (input_audit(args)[0] != audit or model_proof(args) != model or environment(args) != env
            or read_config(args.config)[1] != config_sha or bounded_read(args.pack, config['maxPackBytes']) != raw_pack
            or node_review.digest(Path(__file__).read_bytes()) != tool_sha
            or node_review.digest(Path(training_inputs.__file__).read_bytes()) != verifier_sha):
        raise ValueError('inputs changed during prediction')
    rows = [{'schemaVersion': 'lfg.node-prediction.v1', 'itemSha256': row['review']['itemSha256'],
             'inputSha256': row['item']['inputSha256'], 'snapshotSha256': audit['snapshotSha256'], 'choice': choice}
            for row, choice in zip(validation, choices, strict=True)]
    raw = b''.join(node_review.encode(row) + b'\n' for row in rows)
    if len(raw) > config['maxPredictionBytes']:
        raise ValueError('predictions exceed configured byte limit')
    node_review.create_new(args.output, raw)
    return {'schemaVersion': 'lfg.node-exemplar-prediction.v1', 'predictionsSha256': node_review.digest(raw),
            'packSha256': args.pack_sha256, 'configSha256': config_sha, 'items': len(rows),
            'abstained': sum(choice is None for choice in choices),
            'promotionEligible': False}


def probe(args: argparse.Namespace) -> dict:
    tool_sha = node_review.digest(Path(__file__).read_bytes())
    if args.repetitions < 1:
        raise ValueError('probe repetitions must be positive')
    config, config_sha = read_config(args.config)
    items, ledger, ledger_raw = node_review.load_ledger(args.queue, args.snapshot, args.ledger)
    if len(items) > config['maxValidationItems']:
        raise ValueError('probe exceeds configured item limit')
    model = model_proof(args)
    encoder = Encoder(args.model, config)
    vectors, lengths = encoder.encode([item['text'] for item in items])
    first_digest = node_review.digest(node_review.encode(vectors))
    repeat_exact = None
    for _ in range(1, args.repetitions):
        repeated, repeated_lengths = encoder.encode([item['text'] for item in items])
        same = repeated_lengths == lengths and node_review.digest(node_review.encode(repeated)) == first_digest
        repeat_exact = same if repeat_exact is None else repeat_exact and same
    if (model_proof(args) != model or read_config(args.config)[1] != config_sha
            or node_review.digest(Path(__file__).read_bytes()) != tool_sha):
        raise ValueError('model or configuration changed during probe')
    if node_review.load_ledger(args.queue, args.snapshot, args.ledger)[2] != ledger_raw:
        raise ValueError('review ledger changed during probe')
    return {'schemaVersion': 'lfg.node-encoder-probe.v1', 'model': model, 'encoderVersions': encoder.versions,
            'configSha256': config_sha, 'queueSha256': ledger['queueSha256'], 'items': len(items),
            'toolSha256': tool_sha,
            'dimension': len(vectors[0]), 'minTokens': min(lengths), 'maxTokens': max(lengths),
            'repetitions': args.repetitions, 'sameProcessExact': repeat_exact,
            'labelsUsed': False, 'vectorsSaved': False, 'qualityEvidence': False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    for command_name in ('probe', 'build', 'predict'):
        command = sub.add_parser(command_name)
        for name in ('model', 'model-manifest', 'config', 'snapshot'):
            command.add_argument('--' + name, type=Path, required=True)
        command.add_argument('--revision', required=True)
        if command_name == 'probe':
            command.add_argument('--queue', type=Path, required=True)
            command.add_argument('--ledger', type=Path, required=True)
            command.add_argument('--repetitions', type=int, required=True)
        else:
            command.add_argument('--input', type=Path, nargs=2, action='append', required=True, metavar=('QUEUE', 'LEDGER'))
            command.add_argument('--policy', type=Path, required=True)
            command.add_argument('--node', required=True)
            command.add_argument('--environment-lock', type=Path, required=True)
            command.add_argument('--wheelhouse', type=Path, required=True)
            command.add_argument('--allowed-extra', action='append', default=[])
            command.add_argument('--output', type=Path, required=True)
        if command_name == 'predict':
            command.add_argument('--pack', type=Path, required=True)
            command.add_argument('--pack-sha256', required=True)
    args = parser.parse_args()
    result = {'probe': probe, 'build': build, 'predict': predict}[args.command](args)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, json.JSONDecodeError) as error:
        print(f'node exemplars: {error}', file=sys.stderr)
        sys.exit(1)
