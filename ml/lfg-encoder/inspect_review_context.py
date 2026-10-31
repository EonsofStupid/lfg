#!/usr/bin/env python3
"""Inspect a pending record's verified context without changing its workspace."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import corpus_audit
import review_corpus
import review_web


def inspect(workspace: Path, identifier: str, config_path: Path, show_context: bool = False) -> dict:
    if not identifier.strip():
        raise ValueError('a nonempty pending record ID is required')
    config = review_web.load_config(config_path)
    if 'source_verification' not in config or 'source_context' not in config:
        raise ValueError('source verification and context limits must be configured')
    plan, _, ledger = review_corpus.load_workspace(workspace)
    item = ledger.get(identifier)
    if not item or item['decision'] != 'pending':
        raise ValueError('only a pending record can be inspected')
    _, rows, index, source = review_corpus.find_row(workspace, plan, identifier)
    _, taxonomy = review_corpus.evaluate.load_taxonomy(review_corpus.private_path(workspace, plan['taxonomy']))
    digest = corpus_audit.annotation_digest(rows[index], source['project'], source['split'], taxonomy)
    result = review_web.verify_source(workspace, identifier, digest, config['source_verification'], config['source_context'])
    output = {'status': result['status'], 'record_id': identifier, 'prior_messages': len(result['messages']),
              'truncated': result['truncated'], 'review_decision': 'pending'}
    if 'context_scope' in result:
        output['context_scope'] = result['context_scope']
    if show_context:
        output['messages'] = result['messages']
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--record-id', required=True, help='pending ID shown by the review screen')
    parser.add_argument('--config', type=Path, default=review_web.DEFAULT_CONFIG)
    parser.add_argument('--show-context', action='store_true', help='print private prior messages to the terminal')
    args = parser.parse_args()
    try:
        print(json.dumps(inspect(args.workspace, args.record_id, args.config, args.show_context), ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, TypeError, UnicodeError) as error:
        print(f'context inspection rejected: {error}', file=sys.stderr)
        raise SystemExit(1) from error


if __name__ == '__main__':
    main()
