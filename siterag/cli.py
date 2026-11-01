import argparse
import json

from .config import Settings
from .db import Database
from .index import build_index
from .ingest import collect
from .models import Models
from .search import SearchEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['collect', 'index', 'search', 'models', 'evaluate'])
    parser.add_argument('--limit', type=int, default=100)
    parser.add_argument('--q', default='Что такое искусственный интеллект?')
    parser.add_argument('--mode', choices=['auto', 'never', 'always'], default='auto')
    parser.add_argument('--questions', default='benchmarks/questions.jsonl')
    parser.add_argument('--api-url', help='For evaluate: measure a running API without a second ML process')
    parser.add_argument('--output', help='Evaluation JSON path; default DATA_DIR/reports/results.json')
    args = parser.parse_args()
    cfg = Settings()
    db = Database(cfg.database)
    db.seed()
    models = Models(cfg)
    if args.command == 'collect':
        for source in db.sources(public=True):
            if source['kind'] == 'upload':
                continue
            try:
                print(json.dumps({'source': source['name'], **collect(db, source, args.limit)}, ensure_ascii=False), flush=True)
            except Exception as error:
                print(json.dumps({'source': source['name'], 'error': str(error)}, ensure_ascii=False), flush=True)
    elif args.command == 'index':
        print(json.dumps(build_index(db, models, cfg), ensure_ascii=False, indent=2))
    elif args.command == 'models':
        for kind in ('encoder', 'gate', 'reranker'):
            models.get(kind)
            print(f'{kind}: ready', flush=True)
        print(json.dumps(models.status(), ensure_ascii=False))
    elif args.command == 'evaluate':
        from .evaluation import evaluate, HTTPBenchmarkEngine
        engine = HTTPBenchmarkEngine(args.api_url, cfg) if args.api_url else SearchEngine(db, models, cfg)
        print(json.dumps(evaluate(engine, args.questions, args.output), ensure_ascii=False, indent=2))
    else:
        print(json.dumps(SearchEngine(db, models, cfg).search(args.q, rerank=args.mode), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
