#!/usr/bin/env python3
"""Advisory context only; never store or echo user prompts."""
import json
import sys


def respond(event):
    name = event.get('hook_event_name')
    if name == 'UserPromptSubmit' and 'clyffy-grounded-delivery' in str(event.get('prompt', '')).lower():
        message = 'The Clyffy grounded-delivery skill was requested. Read its SKILL.md, honor user instructions, validate the handoff and use the configured routes. Hooks do not switch models.'
    elif name == 'SubagentStart' and event.get('agent_type') in ('clyffy_planner', 'clyffy_executor'):
        message = 'Follow the Clyffy grounded-delivery skill. Require a validated, current handoff; preserve the user scope and report actual evidence.'
    else:
        return {}
    return {'hookSpecificOutput': {'hookEventName': name, 'additionalContext': message}}


if __name__ == '__main__':
    try:
        raw = sys.stdin.buffer.read(262145)
        event = json.loads(raw) if len(raw) <= 262144 else {}
        print(json.dumps(respond(event) if isinstance(event, dict) else {}))
    except (ValueError, UnicodeError):
        print('{}')
