"""Tests for the comparison harness with stand-in contestants and synthetic rows; no model is called."""
import json
import os
import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import harness_run  # noqa: E402
import harness_score  # noqa: E402


def rows():
    out = []
    for i in range(40):
        level = i % 5
        out.append(dict(id=f'r{i}', src='claude' if i % 2 else 'codex', project=f'p{i % 4}', text=f'request {i} about module{i}',
                        tools=[1, 5, 20, 60, 150][level], level=level, intent=harness_run.INTENTS[i % 5], later=i >= 28,
                        next_frustrated=(i % 3 == 0), next_text=f'please also update module{i} documentation'))
    return out


class Perfect:
    def __init__(self, spec):
        self.truth = {r['id']: r for r in rows()}

    def answer(self, row, _):
        truth = self.truth[row['id']]
        return dict(work_level=truth['level'], intent=truth['intent'], clarification_risk=1.0 if truth['next_frustrated'] else 0.0,
                    missing=['answer'], recipe=dict(context=truth['next_text'], objective='x')), 'raw'


class Broken:
    def __init__(self, spec):
        pass

    def answer(self, row, _):
        raise RuntimeError('model down')


class HarnessTest(unittest.TestCase):
    def test_extract_and_normalize_keep_only_well_formed_fields(self):
        reply = 'Sure.\n```json\n{"work_level": 3, "intent": "change", "clarification_risk": 0.4, "missing": ["answer", "bogus"], "recipe": {"objective": "do it"}}\n```'
        value = harness_run.normalize(harness_run.extract_json(reply))
        self.assertEqual(value['work_level'], 3.0)
        self.assertEqual(value['intent'], 'change')
        self.assertEqual(value['missing'], ['answer'])
        self.assertEqual(value['recipe']['objective'], 'do it')
        self.assertEqual(value['recipe']['tone'], '')
        bad = harness_run.normalize({'work_level': 9, 'intent': 'dance', 'clarification_risk': 2})
        self.assertEqual((bad['work_level'], bad['intent'], bad['clarification_risk']), (None, None, None))
        self.assertIsNone(harness_run.extract_json('no json here'))

    def test_run_is_resumable_and_failures_are_recorded(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness_run.KINDS['perfect'] = Perfect
            harness_run.KINDS['broken'] = Broken
            data = rows()
            text, sha = harness_run.instructions()
            first = harness_run.run_contestant(dict(name='perfect', kind='perfect'), data[:10], tmp, text, sha, 4)
            again = harness_run.run_contestant(dict(name='perfect', kind='perfect'), data, tmp, text, sha, 4)
            self.assertEqual((first['new_ok'], again['already'], again['new_ok']), (10, 10, 30))
            broken = harness_run.run_contestant(dict(name='broken', kind='broken'), data, tmp, text, sha, 2)
            self.assertEqual(broken['new_failed'], 40)
            lines = [json.loads(l) for l in open(os.path.join(tmp, 'broken.jsonl'), encoding='utf-8')]
            self.assertEqual(lines[0]['error'], 'RuntimeError')

    def test_scores_reward_a_perfect_contestant_on_every_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            harness_run.KINDS['perfect'] = Perfect
            data = rows()
            text, sha = harness_run.instructions()
            harness_run.run_contestant(dict(name='perfect', kind='perfect'), data, tmp, text, sha, 4)
            answers = [json.loads(l) for l in open(os.path.join(tmp, 'perfect.jsonl'), encoding='utf-8')]
            report = harness_score.score_contestant({r['id']: r for r in data}, answers)
            self.assertEqual(report['all']['work_level_exact'], 1.0)
            self.assertGreater(report['all']['work_level_rank_corr'][0], 0.95)
            self.assertEqual(report['all']['intent_accuracy'][0], 1.0)
            self.assertEqual(report['all']['clarification_auc'][0], 1.0)
            self.assertGreater(report['all']['recipe_anticipation_mean'], 0.9)
            self.assertEqual(report['later']['work_level_n'], 12)

    def test_auc_and_macro_f1_basics(self):
        self.assertEqual(harness_score.auc([(0.9, True), (0.1, False)]), 1.0)
        self.assertEqual(harness_score.auc([(0.1, True), (0.9, False)]), 0.0)
        self.assertAlmostEqual(harness_score.macro_f1([('change', 'change'), ('run', 'answer')]), 1 / 3)

    def test_the_instructions_ask_for_every_scored_field(self):
        text, _ = harness_run.instructions()
        for field in ('work_level', 'intent', 'clarification_risk', 'missing', 'recipe', *harness_run.INTENTS):
            self.assertIn(field, text)


if __name__ == '__main__':
    unittest.main()
