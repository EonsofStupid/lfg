#!/usr/bin/env python3
"""Install one personal skill, custom agents and advisory hooks without overwriting."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tomllib


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--codex-home', type=Path, default=Path(os.environ.get('CODEX_HOME', Path.home() / '.codex')))
    args = parser.parse_args()
    home = args.codex_home.expanduser().resolve()
    source = Path(__file__).resolve().parents[1]
    dest = home / 'skills' / 'clyffy-grounded-delivery'
    roles = home / 'agents'
    hook_file = home / 'hooks.json'
    targets = [dest, roles / 'clyffy_planner.toml', roles / 'clyffy_executor.toml', hook_file]
    conflicts = [str(p) for p in targets if p.exists()]
    if conflicts:
        raise ValueError('refusing to overwrite existing paths: ' + ', '.join(conflicts))
    if source == dest or dest.is_relative_to(source):
        raise ValueError('installation destination overlaps the skill source')
    with (source / 'routing.toml').open('rb') as stream:
        routes = tomllib.load(stream)
    for stage in ('planner', 'executor'):
        for field in ('model', 'effort'):
            if not isinstance(routes.get(stage, {}).get(field), str) or not routes[stage][field]:
                raise ValueError('routing policy is incomplete')
    shutil.copytree(source, dest, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    roles.mkdir(parents=True, exist_ok=True)
    for stage in ('planner', 'executor'):
        name = 'clyffy_' + stage
        description = ('Ground substantial Clyffy work and produce a verifiable handoff.' if stage == 'planner'
                       else 'Execute one ready Clyffy handoff task with evidence and escalate uncertainty.')
        instructions = ('Read the clyffy-grounded-delivery skill and current user instructions. Ground the plan in current evidence, keep data private and make a validated, scoped handoff.' if stage == 'planner'
                        else 'Read the clyffy-grounded-delivery skill and current user instructions. Validate the current handoff and anchors, work only within the authorized next task, and report actual checks and blockers.')
        fields = {'name': name, 'description': description, 'model': routes[stage]['model'],
                  'model_reasoning_effort': routes[stage]['effort'], 'developer_instructions': instructions}
        with (roles / (name + '.toml')).open('x', encoding='utf8') as stream:
            stream.write(''.join(k + ' = ' + json.dumps(v) + '\n' for k, v in fields.items()))
    command_parts = [sys.executable, str(dest / 'scripts' / 'hook.py')]
    command = subprocess.list2cmdline(command_parts) if os.name == 'nt' else shlex.join(command_parts)
    handler = {'type': 'command', 'command': command, 'timeout': 3, 'additionalContextLimit': 500}
    hooks = {'description': 'Clyffy grounded delivery reminders', 'hooks': {
        'UserPromptSubmit': [{'hooks': [handler]}],
        'SubagentStart': [{'matcher': '^clyffy_(planner|executor)$', 'hooks': [handler]}]}}
    with hook_file.open('x', encoding='utf8') as stream:
        json.dump(hooks, stream, indent=2)
        stream.write('\n')
    print(json.dumps({'skill': str(dest), 'agents': str(roles), 'hooks': str(hook_file),
                      'hook_trust': 'requires host review; not automatically granted'}))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print('Installation failed: ' + str(error), file=sys.stderr)
        raise SystemExit(2) from None
