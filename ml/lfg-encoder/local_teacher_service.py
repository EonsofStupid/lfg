#!/usr/bin/env python3
"""Start or resume a private, finite local-teacher collection job."""
import argparse
import os
from pathlib import Path


def command(python, collector, selection, config, policy, output):
    result = [str(python), str(collector), '--selection', str(selection), '--config', str(config),
              '--policy', str(policy), '--output', str(output)]
    if output.exists():
        result.append('--resume')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('python', 'collector', 'selection', 'config', 'policy', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    paths = (args.python, args.collector, args.selection, args.config, args.policy, args.output)
    if not all(path.is_absolute() for path in paths):
        parser.error('all paths must be absolute')
    argv = command(*paths)
    os.execv(str(args.python), argv)


if __name__ == '__main__':
    main()
