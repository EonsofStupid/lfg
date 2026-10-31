#!/usr/bin/env python3
"""Check installed locked-wheel payloads in the current Python environment."""
import argparse
import json
from pathlib import Path

import training_inputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--lock', type=Path, required=True)
    parser.add_argument('--wheelhouse', type=Path, required=True)
    parser.add_argument('--allowed-extra', action='append', default=[])
    args = parser.parse_args()
    try:
        print(json.dumps(training_inputs.verify_environment(args.lock, args.wheelhouse, args.allowed_extra,
                                                            verify_payload=True), indent=2, allow_nan=False))
    except (OSError, ValueError, KeyError, TypeError, UnicodeError) as error:
        raise SystemExit(f'training environment rejected: {error}') from error


if __name__ == '__main__':
    main()
