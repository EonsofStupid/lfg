#!/usr/bin/env python3
"""Run contestants on the harness prompts: LFG, Claude Code, Codex, OpenRouter or any OpenAI-compatible model.

Every contestant gets the same request and, where it is a language model, the same instructions
(harness_instructions.v1.md). Each returns a work level, an intent, a clarification risk, missing recipe
slots and a CO-STAR-A recipe; LFG returns the parts it has. Answers are appended to one private JSONL file
per contestant, so a run can be stopped and resumed. Prompts and answers never leave the private output
directory except to the contestant itself.

    python ml/lfg-encoder/harness_run.py --config harness.json [--limit N] [--only NAME]

See docs/lfg/TEST-HARNESS.md for the configuration.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
INTENTS = ('answer', 'investigate', 'change', 'run', 'delegate')
SLOTS = ('context', 'objective', 'style', 'tone', 'audience', 'response', 'answer')
# LFG's rule intents mapped onto the action intents the harness scores.
LFG_INTENT = {'build.software': 'change', 'create.image': 'change', 'research.web': 'investigate',
              'evaluate.semantic': 'investigate', 'verify.outcome': 'run', 'respond.general': 'answer'}


def instructions(path=None):
    path = path or os.path.join(HERE, 'harness_instructions.v1.md')
    text = open(path, encoding='utf-8').read()
    return text, hashlib.sha256(text.encode()).hexdigest()


def extract_json(text):
    """The first JSON object in a model's reply, tolerating code fences and prose around it."""
    if not isinstance(text, str):
        return None
    text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip())
    start = text.find('{')
    while start != -1:
        depth = 0
        for end in range(start, len(text)):
            if text[end] == '{':
                depth += 1
            elif text[end] == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:end + 1])
                    except ValueError:
                        break
        start = text.find('{', start + 1)
    return None


def normalize(answer):
    """Keep only well-formed fields; anything malformed becomes None rather than a guess."""
    if not isinstance(answer, dict):
        return dict(work_level=None, intent=None, clarification_risk=None, missing=None, recipe=None)
    level = answer.get('work_level')
    try:
        level = float(level)
        level = level if 0 <= level <= 4 else None
    except (TypeError, ValueError):
        level = None
    intent = answer.get('intent') if answer.get('intent') in INTENTS else None
    risk = answer.get('clarification_risk')
    try:
        risk = float(risk)
        risk = risk if 0 <= risk <= 1 else None
    except (TypeError, ValueError):
        risk = None
    missing = [slot for slot in answer.get('missing') or [] if slot in SLOTS] if isinstance(answer.get('missing'), list) else None
    recipe = answer.get('recipe')
    recipe = {slot: str(recipe.get(slot) or '') for slot in SLOTS} if isinstance(recipe, dict) else None
    return dict(work_level=level, intent=intent, clarification_risk=risk, missing=missing, recipe=recipe)


class Precomputed:
    """Scores made elsewhere, such as LFG's cross-validated work levels: {id, primary} rows."""

    def __init__(self, spec):
        self.scores = {}
        for line in open(spec['path'], encoding='utf-8'):
            row = json.loads(line)
            self.scores[row['id']] = row['primary']

    def answer(self, row, _instructions):
        if row['id'] not in self.scores:
            raise LookupError('no precomputed score for this row')
        return dict(work_level=self.scores[row['id']]), None


class LfgHttp:
    """A running LFG: the assessment's model level when one is bound, else its bracket, and its intent."""

    def __init__(self, spec):
        self.base = spec.get('baseUrl', 'http://127.0.0.1:4317').rstrip('/')
        self.timeout = spec.get('timeoutSeconds', 30)
        # A model trained on these very prompts would be graded on its training data, so it can be ignored.
        self.use_model = spec.get('useModel', True)

    def answer(self, row, _instructions):
        body = json.dumps({'sessionId': 'harness', 'revision': 1, 'text': row['text']}).encode()
        request = urllib.request.Request(self.base + '/api/lfg/assess', data=body, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            value = json.load(response)
        brackets = [step['id'] for step in value['complexity']['ladder']]
        model = value.get('modelScore') or {}
        if self.use_model and model.get('status') == 'available':
            level = model.get('expectedLevel')
        else:
            level = brackets.index(model.get('rulesBracket') or value['complexity']['bracket'])
        return dict(work_level=level, intent=LFG_INTENT.get(value['intent']['primary'])), None


class ClaudeCli:
    """Claude Code in print mode on the owner's subscription, with no tools, hooks or MCP servers."""

    def __init__(self, spec):
        self.binary = spec.get('binary', 'claude')
        self.model = spec.get('model')
        self.timeout = spec.get('timeoutSeconds', 300)
        self.workdir = tempfile.mkdtemp(prefix='lfg-harness-claude-')

    def answer(self, row, instructions_text):
        command = [self.binary, '-p', '--output-format', 'json', '--tools', '', '--strict-mcp-config', '--no-session-persistence',
                   '--settings', json.dumps({'disableAllHooks': True})]
        if self.model:
            command += ['--model', self.model]
        done = subprocess.run(command, input=f'{instructions_text}\n{row["text"]}', capture_output=True, text=True, encoding='utf-8',
                              cwd=self.workdir, timeout=self.timeout)
        if done.returncode != 0:
            raise RuntimeError(f'claude exited {done.returncode}')
        envelope = json.loads(done.stdout)
        return extract_json(envelope.get('result')), envelope.get('result')


class CodexCli:
    """Codex in exec mode on the owner's subscription: read-only, ephemeral, hooks off, in an empty folder."""

    def __init__(self, spec):
        self.binary = spec.get('binary', 'codex')
        self.model = spec.get('model')
        self.timeout = spec.get('timeoutSeconds', 300)
        self.workdir = tempfile.mkdtemp(prefix='lfg-harness-codex-')

    def answer(self, row, instructions_text):
        with tempfile.TemporaryDirectory() as scratch:
            last = os.path.join(scratch, 'last.txt')
            command = [self.binary, 'exec', '--ephemeral', '--skip-git-repo-check', '-s', 'read-only', '-C', self.workdir,
                       '-c', 'features.hooks=false', '--color', 'never', '-o', last]
            if self.model:
                command += ['-m', self.model]
            command.append('-')
            done = subprocess.run(command, input=f'{instructions_text}\n{row["text"]}', capture_output=True, text=True,
                                  encoding='utf-8', timeout=self.timeout)
            if done.returncode != 0 or not os.path.exists(last):
                raise RuntimeError(f'codex exited {done.returncode}')
            text = open(last, encoding='utf-8').read()
        return extract_json(text), text


class OpenAiCompatible:
    """Any OpenAI-compatible chat endpoint, such as OpenRouter or a local llama.cpp or vLLM server.

    The key is read from the environment variable named in the spec and is never stored or logged.
    """

    def __init__(self, spec):
        self.url = spec.get('baseUrl', 'https://openrouter.ai/api/v1').rstrip('/') + '/chat/completions'
        self.model = spec['model']
        self.key_env = spec.get('apiKeyEnv', 'OPENROUTER_API_KEY')
        self.timeout = spec.get('timeoutSeconds', 120)
        self.extra = spec.get('requestFields', {})
        if self.key_env and not os.environ.get(self.key_env):
            raise RuntimeError(f'set {self.key_env} to use this contestant')

    def answer(self, row, instructions_text):
        body = dict(self.extra, model=self.model, messages=[{'role': 'system', 'content': instructions_text}, {'role': 'user', 'content': row['text']}])
        headers = {'Content-Type': 'application/json'}
        if self.key_env:
            headers['Authorization'] = 'Bearer ' + os.environ[self.key_env]
        request = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=headers)
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            value = json.load(response)
        text = value['choices'][0]['message']['content']
        return extract_json(text), text


KINDS = {'precomputed': Precomputed, 'lfg': LfgHttp, 'claude': ClaudeCli, 'codex': CodexCli,
         'openai-compatible': OpenAiCompatible, 'openrouter': OpenAiCompatible}


def run_contestant(spec, rows, out_dir, instructions_text, instructions_sha, workers):
    contestant = KINDS[spec['kind']](spec)
    path = os.path.join(out_dir, f"{spec['name']}.jsonl")
    done = set()
    if os.path.exists(path):
        done = {json.loads(line)['id'] for line in open(path, encoding='utf-8') if line.strip()}
    todo = [row for row in rows if row['id'] not in done]
    lock = threading.Lock()
    counts = dict(ok=0, failed=0)

    def one(row):
        started = time.perf_counter()
        try:
            try:
                answer, raw = contestant.answer(row, instructions_text)
            except LookupError:
                return  # The contestant has nothing for this row, such as later-only scores.
            record = dict(id=row['id'], ok=answer is not None, **normalize(answer if answer is not None else {}))
            if spec['kind'] in ('precomputed', 'lfg'):
                record.update({k: v for k, v in (answer or {}).items() if k in ('work_level', 'intent')})
            record['raw_chars'] = len(raw) if isinstance(raw, str) else None
        except Exception as error:  # noqa: BLE001  one failure must not stop the run
            record = dict(id=row['id'], ok=False, error=type(error).__name__)
        record.update(ms=round((time.perf_counter() - started) * 1000), contestant=spec['name'], instructions_sha256=instructions_sha)
        with lock:
            with open(path, 'a', encoding='utf-8', newline='\n') as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + '\n')
            counts['ok' if record['ok'] else 'failed'] += 1

    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        list(pool.map(one, todo))
    return dict(contestant=spec['name'], already=len(done), new_ok=counts['ok'], new_failed=counts['failed'])


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    parser.add_argument('--config', required=True)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--only', action='append')
    args = parser.parse_args()
    config = json.load(open(args.config, encoding='utf-8'))
    rows = [json.loads(line) for line in open(config['truth'], encoding='utf-8')]
    if config.get('sample') == 'later':
        rows = [row for row in rows if row.get('later')]
    if args.limit:
        rows = rows[:args.limit]
    os.makedirs(config['outDir'], exist_ok=True)
    text, sha = instructions(config.get('instructions'))
    for spec in config['contestants']:
        if args.only and spec['name'] not in args.only:
            continue
        print(json.dumps(run_contestant(spec, rows, config['outDir'], text, sha, spec.get('workers', 1))), flush=True)


if __name__ == '__main__':
    sys.exit(main())
