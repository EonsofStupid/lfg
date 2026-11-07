#!/usr/bin/env python3
"""Export a trained Laya-style decision model as an LFG level-score model directory.

The score question and its options never change, so they become a fixed prefix
(`input.json`), and the graph takes only `input_ids` and `attention_mask` and
returns the level probabilities. The head's attention is written out so the
token count stays dynamic, and weights are stored as fp16 with casts back to
fp32, so arithmetic stays fp32 at half the file size. The export is checked
against the original model before anything is written.

Run it offline, with the network blocked: it only reads local files.

    python ml/lfg-encoder/export_level_score.py --trained DIR --laya DIR --output DIR \
        [--check-texts FILE.jsonl] [--id NAME]

`--trained` holds `model.safetensors`, `encoder/`, `tokenizer/`,
`rl_agent_config.json`, `question.json` and the training `manifest.json`.
`--laya` is the directory with Laya's `rl_common.py`. Outputs and check texts
stay local; this prints aggregates only.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile

os.environ['ORT_DISABLE_TELEMETRY'] = '1'  # Set before import, or model names and hashes go to Microsoft.
os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', HF_DATASETS_OFFLINE='1', DO_NOT_TRACK='1',
                  HF_HUB_DISABLE_TELEMETRY='1', TOKENIZERS_PARALLELISM='false')

import numpy as np  # noqa: E402
import onnx  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402
from onnx import helper, numpy_helper  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

# The fused fast path of nn.TransformerEncoderLayer has no ONNX form; ordinary ops export.
torch.backends.mha.set_fastpath_enabled(False)

MAX_TOKENS = 512
LEVELS = ['0-2', '3-9', '10-30', '31-100', '>100']
SAMPLES = ['ok', 'continue', 'Fix the typo in the README heading.',
           'Add a retry with backoff to the webhook sender and cover it with a test.',
           'Migrate the whole service from REST to gRPC, update every client, and add end-to-end tests across the pipeline.']


def sha(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


class HeadLayer(torch.nn.Module):
    """nn.TransformerEncoderLayer with attention written out. Batch is 1; the token count is dynamic."""

    def __init__(self, layer):
        super().__init__()
        attention = layer.self_attn
        if not (attention.batch_first and attention._qkv_same_embed_dim and attention.in_proj_bias is not None):
            raise SystemExit('unsupported head attention layout')
        self.layer = layer
        self.heads = attention.num_heads
        self.width = attention.embed_dim // attention.num_heads

    def attend(self, x, keep):
        attention = self.layer.self_attn
        q, k, v = torch.nn.functional.linear(x, attention.in_proj_weight, attention.in_proj_bias).chunk(3, -1)
        q, k, v = (part.reshape(1, -1, self.heads, self.width).transpose(1, 2) for part in (q, k, v))
        out = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=keep)
        return attention.out_proj(out.transpose(1, 2).reshape(1, -1, self.heads * self.width))

    def forward(self, x, keep):
        layer = self.layer

        def feed(y):
            return layer.linear2(layer.activation(layer.linear1(y)))

        if layer.norm_first:
            x = x + self.attend(layer.norm1(x), keep)
            return x + feed(layer.norm2(x))
        x = layer.norm1(x + self.attend(x, keep))
        return layer.norm2(x + feed(x))


class LevelProbabilities(torch.nn.Module):
    """DecisionModel.forward for the fixed score question, returning only the level probabilities."""

    def __init__(self, model, markers, qtype):
        super().__init__()
        self.model = model
        self.layers = torch.nn.ModuleList(HeadLayer(layer) for layer in model.head.layers) if model.head is not None else torch.nn.ModuleList()
        self.register_buffer('markers', torch.tensor(markers, dtype=torch.long))
        self.register_buffer('qtype', torch.tensor([qtype], dtype=torch.long))

    def forward(self, input_ids, attention_mask):
        h = self.model.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        h = h + self.model.type_emb(self.qtype)[:, None, :]
        keep = attention_mask.bool()[:, None, None, :]
        for layer in self.layers:
            h = layer(h, keep)
        logits = self.model.scorer(h[:, self.markers, :]).squeeze(-1).float()
        return torch.softmax(logits[:, :len(LEVELS)], -1)


def store_fp16(source, target, minimum=1024):
    """Store large fp32 weights as fp16 and cast them back, so arithmetic stays fp32."""
    model = onnx.load(source)
    casts = []
    for init in list(model.graph.initializer):
        if init.data_type != onnx.TensorProto.FLOAT or int(np.prod(init.dims)) < minimum:
            continue
        half = numpy_helper.from_array(numpy_helper.to_array(init).astype(np.float16), init.name + '__fp16')
        model.graph.initializer.remove(init)
        model.graph.initializer.append(half)
        casts.append(helper.make_node('Cast', [half.name], [init.name], to=onnx.TensorProto.FLOAT, name=init.name + '__cast'))
    for node in reversed(casts):
        model.graph.node.insert(0, node)
    onnx.save(model, target)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--trained', required=True)
    parser.add_argument('--laya', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--check-texts', help='JSONL with a "text" field, for the parity check')
    parser.add_argument('--id', help='model id; defaults to the trained directory name')
    args = parser.parse_args()
    sys.path.insert(0, args.laya)
    from rl_common import QTYPES, build_model, build_sequence

    trained = args.trained
    model_id = args.id or os.path.basename(os.path.normpath(trained))
    config = json.load(open(os.path.join(trained, 'rl_agent_config.json'), encoding='utf-8'))
    question = json.load(open(os.path.join(trained, 'question.json'), encoding='utf-8'))
    tokenizer = AutoTokenizer.from_pretrained(os.path.join(trained, 'tokenizer'), local_files_only=True)
    empty, markers = build_sequence(tokenizer, '', question, MAX_TOKENS, config['head_max_len'])
    if empty[-1] != tokenizer.sep_token_id or len(markers) != len(LEVELS):
        raise SystemExit('unexpected question layout')
    prefix = empty[:-1]
    room = MAX_TOKENS - len(prefix) - 1

    def ids_for(text):
        body = tokenizer(text.replace(tokenizer.mask_token, ' '), add_special_tokens=False)['input_ids'][:room]
        return prefix + body + [tokenizer.sep_token_id]

    texts = list(SAMPLES)
    if args.check_texts:
        rows = [json.loads(line) for line in open(args.check_texts, encoding='utf-8')]
        texts += [row['text'] for row in rows][::max(1, len(rows) // 200)]
    mismatched = sum(ids_for(t) != build_sequence(tokenizer, t, question, MAX_TOKENS, config['head_max_len'])[0] for t in texts)
    if mismatched:
        raise SystemExit(f'{mismatched} texts build different ids from the original model')

    model = build_model(config, encoder_dir=os.path.join(trained, 'encoder'))
    model.load_state_dict(load_file(os.path.join(trained, 'model.safetensors')), strict=True)
    model.encoder.config.reference_compile = False
    model.eval()
    wrapper = LevelProbabilities(model, markers, QTYPES['score']).eval()

    def reference(text):
        ids = ids_for(text)
        logits, _ = model(torch.tensor([ids]), torch.ones(1, len(ids), dtype=torch.long), torch.tensor([markers]),
                          torch.ones(1, len(LEVELS), dtype=torch.bool), torch.tensor([QTYPES['score']]))
        return torch.softmax(logits[0, :len(LEVELS)].float(), -1).numpy()

    with torch.no_grad():
        expected = np.array([reference(t) for t in texts])
        rewritten = np.array([wrapper(torch.tensor([ids_for(t)]), torch.ones(1, len(ids_for(t)), dtype=torch.long)).numpy()[0] for t in texts])
    rewrite_error = float(np.abs(rewritten - expected).max())
    if rewrite_error > 1e-4:
        raise SystemExit(f'head rewrite differs from the original model by {rewrite_error}')

    os.makedirs(args.output, exist_ok=True)
    with tempfile.TemporaryDirectory() as scratch:
        fp32 = os.path.join(scratch, 'model.fp32.onnx')
        sample = torch.tensor([ids_for(SAMPLES[3])])
        with torch.no_grad():
            torch.onnx.export(wrapper, (sample, torch.ones_like(sample)), fp32, input_names=['input_ids', 'attention_mask'],
                              output_names=['level_probabilities'], dynamic_axes={'input_ids': {1: 'tokens'}, 'attention_mask': {1: 'tokens'}},
                              opset_version=17, do_constant_folding=True, dynamo=False, external_data=False)
        target = os.path.join(args.output, 'model.onnx')
        store_fp16(fp32, target)
    session = ort.InferenceSession(target, providers=['CPUExecutionProvider'])
    exported = np.array([session.run(None, {'input_ids': np.array([ids_for(t)], dtype=np.int64),
                                            'attention_mask': np.ones((1, len(ids_for(t))), dtype=np.int64)})[0][0] for t in texts])
    levels = np.arange(len(LEVELS))
    level_error = float(np.abs(exported @ levels - expected @ levels).max())
    if level_error > 0.01:
        raise SystemExit(f'exported model differs from the original by {level_error} levels')

    shutil.copy(os.path.join(trained, 'tokenizer', 'tokenizer.json'), os.path.join(args.output, 'tokenizer.json'))
    input_path = os.path.join(args.output, 'input.json')
    with open(input_path, 'w', encoding='utf-8', newline='\n') as handle:
        json.dump(dict(schema='lfg.dev-model-input.v1', prefix_ids=prefix, marker_positions=markers, max_tokens=MAX_TOKENS,
                       room=room, sep_id=tokenizer.sep_token_id, pad_id=tokenizer.pad_token_id, mask_token=tokenizer.mask_token,
                       output='level_probabilities[5]; score = sum(p[i] * i)', levels=LEVELS,
                       tokenizer_sha256=sha(os.path.join(args.output, 'tokenizer.json'))), handle)
    training_path = os.path.join(trained, 'manifest.json')
    training = json.load(open(training_path, encoding='utf-8')) if os.path.exists(training_path) else {}
    manifest = {
        'schemaVersion': 'lfg.model-score-manifest.v1', 'id': model_id, 'promotionEligible': False,
        'modelFile': 'model.onnx', 'modelSha256': sha(os.path.join(args.output, 'model.onnx')),
        'tokenizerSha256': sha(os.path.join(args.output, 'tokenizer.json')), 'inputSpecSha256': sha(input_path),
        'inputs': ['input_ids', 'attention_mask'], 'output': 'level_probabilities', 'levels': len(LEVELS),
        'provenance': {
            'trainingManifestSha256': sha(training_path) if training else None,
            'base': training.get('base'), 'data': training.get('data'), 'recipe': training.get('recipe'),
            'storage': 'weights stored as fp16 and cast to fp32 at load; arithmetic is fp32',
            'checkedTexts': len(texts), 'maxLevelDifference': round(level_error, 5),
        },
    }
    manifest_path = os.path.join(args.output, 'manifest.json')
    with open(manifest_path, 'w', encoding='utf-8', newline='\n') as handle:
        json.dump(manifest, handle, indent=1)
    print(json.dumps({'id': model_id, 'prefixTokens': len(prefix), 'checkedTexts': len(texts),
                      'maxLevelDifference': round(level_error, 5), 'modelBytes': os.path.getsize(os.path.join(args.output, 'model.onnx')),
                      'manifestSha256': sha(manifest_path)}))


if __name__ == '__main__':
    main()
