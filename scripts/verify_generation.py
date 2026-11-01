"""One real LLM request to check refusal on controlled, irrelevant evidence."""
import json
from pathlib import Path

from siterag.config import Settings
from siterag.generation import answer, REFUSAL


def main():
    cfg = Settings()
    if not cfg.api_key or not cfg.api_url:
        raise SystemExit('Set SUB2API_API_KEY and SUB2API_BASE_URL in the process environment')
    question = 'Какой секретный код у двери кабинета QX-4729?'
    empty = answer(question, {'hits': [], 'meta': {'warnings': [], 'timings_ms': {'total': 0}}}, cfg)
    irrelevant = answer(question, {'hits': [{'title': 'Контрольный источник', 'url': 'https://example.org/control',
                         'text': 'В офисе стоят три деревянных стула. Это все доступные сведения.'}],
                         'meta': {'warnings': [], 'timings_ms': {'total': 0}}}, cfg)
    assert empty['answer'] == REFUSAL
    assert irrelevant['answer_mode'] == 'llm' and irrelevant['answer'].strip() == REFUSAL, irrelevant['answer']
    report = {'model': irrelevant['model'], 'empty_sources_refusal': True,
              'irrelevant_sources_refusal': True, 'irrelevant_sources_mode': 'llm',
              'answer': irrelevant['answer'], 'latency_ms': irrelevant['meta']['timings_ms']['generation']}
    Path('benchmarks/generation-checks.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
