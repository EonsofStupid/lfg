#!/usr/bin/env python3
"""Write the tiny synthetic model-score fixture used by lfg-server's native tests.

The graph is not a trained model. It turns the mean token id and the token count
into five level probabilities, so a test can recompute the exact answer and so
check tokenization, prefix assembly, truncation and digests through tract.
Needs the onnx package. argv: output directory.
"""
import hashlib
import json
import os
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)

LEVEL_WEIGHTS = [-0.1, -0.05, 0.0, 0.05, 0.1]
COUNT_WEIGHT = 0.01
VOCAB = ['[UNK]', '[SEP]', '[MASK]', 'fix', 'login', 'add', 'tests', 'across', 'every', 'service']


def sha(path):
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def write_json(name, value):
    path = os.path.join(OUT, name)
    with open(path, 'w', encoding='utf-8', newline='\n') as handle:
        json.dump(value, handle, indent=1, sort_keys=True)
        handle.write('\n')
    return path


ids = helper.make_tensor_value_info('input_ids', TensorProto.INT64, [1, 'S'])
mask = helper.make_tensor_value_info('attention_mask', TensorProto.INT64, [1, 'S'])
out = helper.make_tensor_value_info('level_probabilities', TensorProto.FLOAT, [1, 5])
weights = numpy_helper.from_array(np.array([LEVEL_WEIGHTS], dtype=np.float32), 'level_weights')
count_weight = numpy_helper.from_array(np.full((1, 5), COUNT_WEIGHT, dtype=np.float32), 'count_weights')
nodes = [
    helper.make_node('Cast', ['input_ids'], ['ids_f'], to=TensorProto.FLOAT),
    helper.make_node('ReduceMean', ['ids_f'], ['mean'], axes=[1], keepdims=1),
    helper.make_node('Cast', ['attention_mask'], ['mask_f'], to=TensorProto.FLOAT),
    helper.make_node('ReduceSum', ['mask_f', 'axes'], ['count'], keepdims=1),
    helper.make_node('MatMul', ['mean', 'level_weights'], ['by_mean']),
    helper.make_node('MatMul', ['count', 'count_weights'], ['by_count']),
    helper.make_node('Add', ['by_mean', 'by_count'], ['logits']),
    helper.make_node('Softmax', ['logits'], ['level_probabilities'], axis=-1),
]
axes = numpy_helper.from_array(np.array([1], dtype=np.int64), 'axes')
graph = helper.make_graph(nodes, 'level-score-fixture', [ids, mask], [out], [weights, count_weight, axes])
model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 17)], producer_name='lfg-level-score-fixture')
model.ir_version = 8
onnx.checker.check_model(model)
model_path = os.path.join(OUT, 'model.onnx')
onnx.save(model, model_path)

special = [dict(id=i, content=VOCAB[i], single_word=False, lstrip=False, rstrip=False, normalized=False, special=True) for i in range(3)]
tokenizer_path = write_json('tokenizer.json', {
    'version': '1.0', 'truncation': None, 'padding': None, 'added_tokens': special,
    'normalizer': {'type': 'Lowercase'}, 'pre_tokenizer': {'type': 'Whitespace'},
    'post_processor': None, 'decoder': None,
    'model': {'type': 'WordLevel', 'vocab': {word: i for i, word in enumerate(VOCAB)}, 'unk_token': '[UNK]'},
})
prefix = [2, 2, 2, 2, 2, 1]
room = 4
input_path = write_json('input.json', {
    'schema': 'lfg.dev-model-input.v1', 'prefix_ids': prefix, 'marker_positions': [0, 1, 2, 3, 4],
    'max_tokens': len(prefix) + room + 1, 'room': room, 'sep_id': 1, 'pad_id': 0, 'mask_token': '[MASK]',
    'output': 'level_probabilities[5]; score = sum(p[i] * i)', 'levels': ['0-2', '3-9', '10-30', '31-100', '>100'],
    'tokenizer_sha256': sha(tokenizer_path),
})
write_json('manifest.json', {
    'schemaVersion': 'lfg.model-score-manifest.v1', 'id': 'level-score-fixture', 'promotionEligible': False,
    'modelFile': 'model.onnx', 'modelSha256': sha(model_path), 'tokenizerSha256': sha(tokenizer_path),
    'inputSpecSha256': sha(input_path), 'inputs': ['input_ids', 'attention_mask'], 'output': 'level_probabilities',
    'levels': 5, 'provenance': {'kind': 'synthetic test fixture; not a trained model',
                                'levelWeights': LEVEL_WEIGHTS, 'countWeight': COUNT_WEIGHT},
})
print(json.dumps({'manifestSha256': sha(os.path.join(OUT, 'manifest.json')), 'modelBytes': os.path.getsize(model_path)}))
