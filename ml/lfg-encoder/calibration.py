"""Aggregate uncalibrated multi-label sigmoid scores against reviewed labels."""
from __future__ import annotations

import bisect
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import evaluate


def load_config(path: Path) -> tuple[list[float], str]:
    raw = path.read_bytes()
    config = json.loads(raw)
    if not isinstance(config, dict) or set(config) != {'schema_version', 'bin_edges'} or \
            config['schema_version'] != 'lfg.calibration-config.v1':
        raise ValueError('invalid calibration configuration')
    edges = config['bin_edges']
    if (not isinstance(edges, list) or len(edges) < 2
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in edges)
            or edges[0] != 0 or edges[-1] != 1
            or any(a >= b for a, b in zip(edges, edges[1:]))):
        raise ValueError('calibration bin edges must strictly increase from zero to one')
    return [float(value) for value in edges], hashlib.sha256(raw).hexdigest()


class Accumulator:
    def __init__(self, edges: list[float]):
        self.edges = edges
        self.bins = [{'count': 0, 'score_sum': 0.0, 'positives': 0} for _ in range(len(edges) - 1)]
        self.count = 0
        self.squared_error = 0.0

    def add(self, score: float, actual: int) -> None:
        index = min(bisect.bisect_right(self.edges, score) - 1, len(self.bins) - 1)
        bucket = self.bins[index]
        bucket['count'] += 1
        bucket['score_sum'] += score
        bucket['positives'] += actual
        self.count += 1
        self.squared_error += (score - actual) ** 2

    def result(self) -> dict[str, Any]:
        if not self.count:
            raise ValueError('calibration observations must be nonempty')
        results = []
        errors = []
        for index, bucket in enumerate(self.bins):
            count = bucket['count']
            mean = bucket['score_sum'] / count if count else None
            frequency = bucket['positives'] / count if count else None
            if count:
                errors.append(count * abs(mean - frequency))
            results.append({'lower': self.edges[index], 'upper': self.edges[index + 1], 'count': count,
                            'mean_score': mean, 'positive_rate': frequency})
        return {'observations': self.count, 'ece': math.fsum(errors) / self.count,
                'brier': self.squared_error / self.count, 'bins': results}


def calculate(rows: list[dict], predictions: dict[str, dict], taxonomy: dict[str, set[str]],
              decision: dict, edges: list[float]) -> dict[str, Any]:
    universe = set.union(*taxonomy.values())
    if not rows or {row['id'] for row in rows} != set(predictions):
        raise ValueError('calibration IDs do not match the reviewed test')
    by_label = {label: Accumulator(edges) for label in universe}
    pooled = Accumulator(edges)
    for row in rows:
        prediction = predictions[row['id']]
        scores = prediction.get('scores')
        if (not isinstance(scores, dict) or set(scores) != {'semantics', 'values'}
                or scores['semantics'] != 'sigmoid-uncalibrated-v1' or not isinstance(scores['values'], dict)
                or set(scores['values']) != universe):
            raise ValueError('native scores do not cover the complete taxonomy')
        values = scores['values']
        if any(type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
               for value in values.values()):
            raise ValueError('native scores must be finite numbers between zero and one')
        chosen = {label for label, value in values.items()
                  if value >= decision['label_thresholds'].get(label, decision['default_threshold'])}
        for field in evaluate.FIELDS:
            if evaluate.labels(prediction, field) != chosen & taxonomy[field]:
                raise ValueError('native score thresholds disagree with emitted labels')
            actual = evaluate.labels(row, field)
            for label in sorted(taxonomy[field]):
                score, target = float(values[label]), int(label in actual)
                by_label[label].add(score, target)
                pooled.add(score, target)
    summaries = {label: accumulator.result() for label, accumulator in sorted(by_label.items())}
    return {'score_semantics': 'sigmoid-uncalibrated-v1',
            'method': 'specified-bin empirical binary ECE and Brier',
            'intervals': 'left-closed/right-open except final bin includes one',
            'macro_label_ece': math.fsum(item['ece'] for item in summaries.values()) / len(summaries),
            'micro_label_observation': pooled.result(), 'labels': summaries}
