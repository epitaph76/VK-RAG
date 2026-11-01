"""Read-only to the reference corpus; re-ingest one real article into a separate test DB."""
import json
import argparse
from dataclasses import replace
from pathlib import Path

from siterag.config import Settings
from siterag.db import Database
from siterag.ingest import collect
from siterag.ingest import FetchError
from siterag.index import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='https://habr.com/ru/articles/1102362/')
    args = parser.parse_args()
    cfg = Settings()
    reference = Database(cfg.database)
    doc = next(d for d in reference.documents() if d['url'] == args.url)
    test_dir = Path('.local/live-verification')
    db = Database(test_dir / 'test.sqlite3')
    source = next((s for s in db.sources() if s['url'] == doc['url']), None) or db.add_source('Live verification', 'html', doc['url'])
    try:
        first = collect(db, source, limit=1)
        before = db.revision()
        second = collect(db, source, limit=1)
    except FetchError as error:
        report = {'url': doc['url'], 'verified': False, 'error': str(error), 'reference_corpus_unchanged': True}
        atomic_json(Path('benchmarks/live-checks.json'), report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        raise SystemExit(1) from None
    assert second['fetched'] == 1 and second['changed'] == 0 and db.revision() == before
    persisted = len(Database(db.path).documents())
    assert persisted >= 1
    report = {'url': doc['url'], 'verified': True, 'first': first, 'repeat': second, 'persisted_documents': persisted,
              'reference_corpus_unchanged': True}
    atomic_json(Path('benchmarks/live-checks.json'), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
