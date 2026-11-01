import json
import math
import statistics
import time
from pathlib import Path

from .index import atomic_json


def metrics(hits, qrels):
    ranked = list(dict.fromkeys(h['document_id'] for h in hits))
    positives = {doc: int(grade) for doc, grade in qrels.items() if grade > 0}
    if not positives:
        raise ValueError('No positive judgments')
    out = {}
    for k in (5, 10):
        out[f'Recall@{k}'] = sum(d in positives for d in ranked[:k]) / len(positives)
        dcg = sum((2 ** positives.get(doc, 0) - 1) / math.log2(i + 2) for i, doc in enumerate(ranked[:k]))
        ideal = sum((2 ** grade - 1) / math.log2(i + 2) for i, grade in enumerate(sorted(positives.values(), reverse=True)[:k]))
        out[f'nDCG@{k}'] = dcg / ideal
    out['MRR@10'] = next((1 / (i + 1) for i, doc in enumerate(ranked[:10]) if doc in positives), 0.0)
    return out


def evaluate(engine, questions_file, output='benchmarks/results.json'):
    questions = [json.loads(line) for line in Path(questions_file).read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if len(questions) != 60 or sum(q['split'] == 'dev' for q in questions) != 20 or sum(q['split'] == 'test' for q in questions) != 40:
        raise ValueError('Expected 20 dev + 40 test queries')
    dev_docs = {doc for q in questions if q['split'] == 'dev' for doc in q['qrels']}
    test_docs = {doc for q in questions if q['split'] == 'test' for doc in q['qrels']}
    if dev_docs.intersection(test_docs):
        raise ValueError('Document leakage across splits')
    snapshot = engine.current()
    corpus_hash = snapshot.manifest['corpus_hash']
    rows, cold = [], {}
    sample = questions[0].get('query') or questions[0].get('question') or questions[0].get('q')
    for mode in ('never', 'always', 'auto'):
        first = engine.search(sample, rerank=mode, limit=20)
        cold[mode] = first['meta']['timings_ms']
    for number, question in enumerate(questions, 1):
        q = question.get('query') or question.get('question') or question.get('q')
        for mode in ('never', 'always', 'auto'):
            if engine.current().manifest['corpus_hash'] != corpus_hash:
                raise ValueError('Corpus changed during benchmark')
            result = engine.search(q, rerank=mode, limit=20)
            rows.append({'query_id': question['id'], 'split': question['split'], 'mode': mode,
                         **metrics(result['hits'], question['qrels']), 'latency_ms': result['meta']['timings_ms']['total'],
                         'document_ids': list(dict.fromkeys(h['document_id'] for h in result['hits'])), **result['meta']})
        print(f'Benchmark {number}/{len(questions)}', flush=True)
    summary = []
    for split in ('dev', 'test'):
        for mode in ('never', 'always', 'auto'):
            selected = [r for r in rows if r['split'] == split and r['mode'] == mode]
            latencies = sorted(r['latency_ms'] for r in selected)
            paired = {r['query_id']: r for r in rows if r['split'] == split and r['mode'] == 'always'}
            baseline = {r['query_id']: r for r in rows if r['split'] == split and r['mode'] == 'never'}
            useful = [r for r in selected if paired[r['query_id']]['nDCG@5'] > baseline[r['query_id']]['nDCG@5'] + 1e-9]
            summary.append({'split': split, 'mode': mode, 'queries': len(selected),
                            **{name: statistics.mean(r[name] for r in selected) for name in ('Recall@5','Recall@10','nDCG@5','nDCG@10','MRR@10')},
                            'p50_ms': statistics.median(latencies), 'p95_ms': latencies[math.ceil(.95 * len(latencies)) - 1],
                            'rerank_rate': sum(r['rerank_applied'] for r in selected) / len(selected),
                            'confident_keep_rate': sum(r['gate']['reason'] == 'confident_keep' for r in selected) / len(selected),
                            'fallback_rate': sum(bool(r['warnings']) for r in selected) / len(selected),
                            'useful_rerank_queries': len(useful),
                            'missed_useful_rerank': sum(not r['rerank_applied'] for r in useful) if mode == 'auto' else None})
    report = {'generated_at': __import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
              'corpus_hash': corpus_hash, 'index_version': snapshot.manifest['version'],
              'model_revisions': engine.models.revisions, 'threshold': engine.settings.gate_threshold,
              'cold_start_timings_ms': cold, 'summary': summary, 'rows': rows,
              'limitations': ['Agent-authored source-anchored questions; no independent human validation.',
                              'Positive judgments are incomplete; scores measure known reference documents, not exhaustive relevance.',
                              'Small current-source corpus; no claim of generalization.',
                              'Warm latency excludes initial loading; cold-start measurements shown separately.',
                              'Threshold fixed at 0.8 in advance; test set not used for tuning.']}
    atomic_json(Path(output), report)
    return {k: v for k, v in report.items() if k != 'rows'}
