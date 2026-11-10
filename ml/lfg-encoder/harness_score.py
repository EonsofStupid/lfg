#!/usr/bin/env python3
"""Score harness answers against what actually happened. Prints aggregates only.

Jobs, each scored on all prompts and on later prompts, with 95% intervals that resample whole projects:
- work level: rank correlation with realized tool calls (ranks averaged within each tool, then pooled),
  plus exact and within-one level accuracy;
- intent: accuracy and macro-F1 against the action intent read from the tools the agent used;
- clarification risk: AUC for predicting that the person's next prompt was frustrated;
- recipe anticipation: the share of new content words in the person's next prompt that the recipe
  already raised. This is a rough proxy; blind review by the owner is the real test of a recipe.

    python ml/lfg-encoder/harness_score.py --config harness.json [--json report.json]
"""
import argparse
import collections
import json
import math
import os
import random
import re

INTENTS = ('answer', 'investigate', 'change', 'run', 'delegate')
WORD = re.compile(r"[a-z][a-z0-9_.-]{3,}")
STOP = set('this that with from have will would should could there their they them then than what when where which while into onto about '
           'also just like make sure need want please lets dont does done here more some your yours very much only each every same other '
           'been being were okay yeah fine good thing things stuff work working still again'.split())


def ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    out, start = [0.0] * len(values), 0
    while start < len(order):
        end = start
        while end + 1 < len(order) and values[order[end + 1]] == values[order[start]]:
            end += 1
        for position in range(start, end + 1):
            out[order[position]] = (start + end) / 2
        start = end + 1
    return out


def pearson(xs, ys):
    if len(xs) < 3:
        return float('nan')
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    top = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    bottom = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
    return top / bottom if bottom else float('nan')


def rank_corr(pairs):
    """pairs: (prediction, tools, source). Ranks within each source, then one pooled correlation."""
    xs, ys = [], []
    for source in sorted({p[2] for p in pairs}):
        group = [p for p in pairs if p[2] == source]
        if len(group) < 3:
            continue
        xs += [r / max(len(group) - 1, 1) for r in ranks([p[0] for p in group])]
        ys += [r / max(len(group) - 1, 1) for r in ranks([p[1] for p in group])]
    return pearson(xs, ys)


def auc(pairs):
    """pairs: (score, label)."""
    positives = [s for s, y in pairs if y]
    negatives = [s for s, y in pairs if not y]
    if not positives or not negatives:
        return float('nan')
    r = ranks([s for s, _ in pairs])
    pos_ranks = sum(r[i] + 1 for i, (_, y) in enumerate(pairs) if y)
    return (pos_ranks - len(positives) * (len(positives) + 1) / 2) / (len(positives) * len(negatives))


def macro_f1(pairs):
    scores = []
    for label in INTENTS:
        tp = sum(1 for p, t in pairs if p == label and t == label)
        fp = sum(1 for p, t in pairs if p == label and t != label)
        fn = sum(1 for p, t in pairs if p != label and t == label)
        if tp + fp + fn:
            scores.append(2 * tp / (2 * tp + fp + fn))
    return sum(scores) / len(scores) if scores else float('nan')


def anticipation(row, answer):
    recipe = answer.get('recipe')
    if not recipe or not row.get('next_text'):
        return None
    said = set(WORD.findall(row['text'].lower()))
    wanted = {w for w in WORD.findall(row['next_text'].lower()) if w not in said and w not in STOP}
    if not wanted:
        return None
    offered = set(WORD.findall(' '.join(recipe.values()).lower()))
    return len(wanted & offered) / len(wanted)


def with_interval(rows, metric, draws=400, seed=7):
    """Point estimate and a 95% interval resampling whole projects."""
    point = metric(rows)
    by_project = collections.defaultdict(list)
    for row in rows:
        by_project[row[0]['project']].append(row)
    names = sorted(by_project)
    rng = random.Random(seed)
    samples = []
    for _ in range(draws):
        value = metric([r for n in rng.choices(names, k=len(names)) for r in by_project[n]])
        if not math.isnan(value):
            samples.append(value)
    samples.sort()
    if math.isnan(point) or not samples:
        return None
    return [round(point, 3), round(samples[int(.025 * len(samples))], 3), round(samples[int(.975 * len(samples))], 3)]


def score_contestant(truth, answers):
    joined = [(truth[a['id']], a) for a in answers if a['id'] in truth]
    report = dict(answered=len(joined), failed=sum(1 for _, a in joined if not a.get('ok')))
    ms = sorted(a['ms'] for _, a in joined if a.get('ok') and a.get('ms') is not None)
    if ms:
        report['ms_p50'] = ms[len(ms) // 2]
        report['ms_p95'] = ms[int(.95 * (len(ms) - 1))]
    for group, keep in (('all', lambda r: True), ('later', lambda r: r.get('later'))):
        rows = [(r, a) for r, a in joined if keep(r) and a.get('ok')]
        level = [(r, a) for r, a in rows if a.get('work_level') is not None]
        intent = [(r, a) for r, a in rows if a.get('intent') and r.get('intent')]
        risk = [(r, a) for r, a in rows if a.get('clarification_risk') is not None and r.get('next_frustrated') is not None]
        recipes = [(r, a, anticipation(r, a)) for r, a in rows]
        recipes = [(r, a, v) for r, a, v in recipes if v is not None]
        part = {}
        if level:
            part['work_level_rank_corr'] = with_interval(level, lambda xs: rank_corr([(a['work_level'], r['tools'], r['src']) for r, a in xs]))
            part['work_level_exact'] = round(sum(round(a['work_level']) == r['level'] for r, a in level) / len(level), 3)
            part['work_level_within_one'] = round(sum(abs(round(a['work_level']) - r['level']) <= 1 for r, a in level) / len(level), 3)
            part['work_level_n'] = len(level)
        if intent:
            part['intent_accuracy'] = with_interval(intent, lambda xs: sum(a['intent'] == r['intent'] for r, a in xs) / len(xs) if xs else float('nan'))
            part['intent_macro_f1'] = with_interval(intent, lambda xs: macro_f1([(a['intent'], r['intent']) for r, a in xs]))
            part['intent_n'] = len(intent)
        if risk:
            part['clarification_auc'] = with_interval(risk, lambda xs: auc([(a['clarification_risk'], r['next_frustrated']) for r, a in xs]))
            part['clarification_n'] = len(risk)
        if recipes:
            part['recipe_anticipation_mean'] = round(sum(v for _, _, v in recipes) / len(recipes), 3)
            part['recipe_n'] = len(recipes)
        report[group] = part
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--config', required=True)
    parser.add_argument('--json')
    args = parser.parse_args()
    config = json.load(open(args.config, encoding='utf-8'))
    truth = {r['id']: r for r in map(json.loads, open(config['truth'], encoding='utf-8'))}
    reports = {}
    for spec in config['contestants']:
        path = os.path.join(config['outDir'], f"{spec['name']}.jsonl")
        if not os.path.exists(path):
            continue
        latest = {}
        for line in open(path, encoding='utf-8'):
            if line.strip():
                answer = json.loads(line)
                latest[answer['id']] = answer
        reports[spec['name']] = dict(score_contestant(truth, list(latest.values())), size=spec.get('size'), kind=spec['kind'])
    for name, report in reports.items():
        print(json.dumps({name: report}))
    if args.json:
        json.dump(reports, open(args.json, 'w', encoding='utf-8'), indent=1)


if __name__ == '__main__':
    main()
