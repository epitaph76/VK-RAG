from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from siterag.config import Settings
from siterag.db import Database
from siterag.generation import answer, valid_citations
from siterag.index import build_index, chunk_documents
from siterag.ingest import FetchError, canonical_url, parse_article, validate_public_url
from siterag.search import SearchEngine, gate_decision


class Tokenizer:
    def encode(self, text, **kwargs):
        return text.split()
    def decode(self, values, **kwargs):
        return ' '.join(values)


class FakeModels:
    def __init__(self, decision='keep', confidence=.9, broken=False):
        self.decision, self.confidence, self.broken = decision, confidence, broken
        self.revisions = {}
        self.calls = []

    def get(self, kind):
        self.calls.append(kind)
        if kind == 'encoder':
            raise RuntimeError('offline')
        if self.broken:
            raise RuntimeError('offline')
        if kind == 'gate':
            return SimpleNamespace(load=lambda name: SimpleNamespace(tok=Tokenizer()), predict=lambda *a, **kw: {'answers': {'action': {'choice': self.decision, 'answer_confidence': self.confidence}}})
        return SimpleNamespace(predict=lambda pairs, **kw: np.arange(len(pairs), dtype=float))

    def status(self):
        return {'loaded': [], 'errors': {}, 'revisions': {}}


@pytest.fixture
def fixture(tmp_path):
    settings = Settings(data_dir=tmp_path, models_enabled=False)
    db = Database(settings.database)
    source = db.add_source('Test', 'html', 'https://example.org/a')
    for i, (text, published) in enumerate([
        ('alpha яблоки исследование красных яблок. Ученые сравнили три красных яблока.', '2026-10-01T12:00:00+00:00'),
        ('alpha апельсины исследование оранжевых апельсинов. Ученые сравнили апельсины.', '2026-11-01T12:00:00+00:00'),
        ('alpha груши исследование зеленых груш. Ученые изучили груши.', None),
    ]):
        db.upsert(source['id'], f'Doc {i}', f'https://example.org/{i}', text, published)
    models = FakeModels()
    build_index(db, models, settings)
    return db, models, settings, source


def test_chunking_overlap_and_final_window():
    docs = [{'id': 'a', 'text': 'one two three four five six seven'}]
    chunks = chunk_documents(docs, Tokenizer(), 4, 1)
    assert [c['text'] for c in chunks] == ['one two three four', 'four five six seven']
    with pytest.raises(ValueError):
        chunk_documents(docs, size=3, overlap=3)


def test_upsert_deduplicates_text_but_preserves_urls(fixture):
    db, models, cfg, source = fixture
    doc = db.documents()[0]
    before = db.revision()
    assert not db.upsert(source['id'], doc['title'], doc['url'], doc['text'], doc['published_at'])
    assert db.revision() == before
    db.upsert(source['id'], 'Alias', 'https://example.org/alias', doc['text'])
    with db.connect() as connection:
        assert connection.execute('SELECT COUNT(*) FROM contents').fetchone()[0] == 3
        assert connection.execute('SELECT COUNT(*) FROM documents').fetchone()[0] == 4
    assert db.upsert(source['id'], doc['title'], doc['url'], doc['text'] + ' updated', doc['published_at'])


def test_filtering_before_selection_and_disabled_sources(fixture):
    db, models, cfg, source = fixture
    engine = SearchEngine(db, models, cfg)
    result = engine.search('alpha', date_from='2026-11-01', date_to='2026-11-01', rerank='never')
    assert len(result['hits']) == 1 and 'апельсины' in result['hits'][0]['text']
    assert not engine.search('alpha', source_id='missing')['hits']
    db.enable(source['id'], False)
    assert not engine.search('alpha')['hits']


@pytest.mark.parametrize('decision,confidence,expected', [('keep', .9, False), ('keep', .7, True), ('rerank', .99, True)])
def test_gate_controls_independent_reranker(fixture, decision, confidence, expected):
    db, _, cfg, _ = fixture
    models = FakeModels(decision, confidence)
    result = SearchEngine(db, models, cfg).search('alpha')
    assert result['meta']['rerank_applied'] == expected
    assert ('reranker' in models.calls) == expected
    if expected:
        assert result['hits'][0]['rerank_score'] >= result['hits'][-1]['rerank_score']


@pytest.mark.parametrize('confidence', [float('nan'), True, None, 1.1])
def test_invalid_gate_confidence_falls_back_to_rerank(fixture, confidence):
    db, _, cfg, _ = fixture
    models = FakeModels(confidence=confidence)
    result = SearchEngine(db, models, cfg).search('alpha')
    assert result['meta']['gate']['decision'] == 'fallback'
    assert result['meta']['rerank_applied']


def test_model_outages_preserve_bm25_order(fixture):
    db, _, cfg, _ = fixture
    engine = SearchEngine(db, FakeModels(broken=True), cfg)
    base = engine.search('alpha', rerank='never')
    result = engine.search('alpha')
    assert [h['id'] for h in result['hits']] == [h['id'] for h in base['hits']]
    assert not result['meta']['rerank_applied'] and len(result['meta']['warnings']) >= 2


def test_atomic_index_publication_and_reload(fixture):
    db, models, cfg, source = fixture
    engine = SearchEngine(db, models, cfg)
    first = engine.search('alpha', rerank='never')
    db.upsert(source['id'], 'new', 'https://example.org/new', 'quantum уникальная квантовая информация')
    assert not engine.search('quantum', rerank='never')['hits']
    build_index(db, models, cfg)
    second = engine.search('quantum', rerank='never')
    assert second['hits'] and first['meta']['index_version'] != second['meta']['index_version']
    (cfg.data_dir / 'indices' / 'unfinished').mkdir()
    assert SearchEngine(db, models, cfg).search('quantum', rerank='never')['hits']


def test_citations_and_extractive_fallback(fixture, monkeypatch):
    db, models, cfg, _ = fixture
    result = SearchEngine(db, models, cfg).search('alpha', rerank='never')
    monkeypatch.setattr('siterag.generation.requests.post', lambda *a, **kw: SimpleNamespace(raise_for_status=lambda: None, json=lambda: {'choices': [{'message': {'content': 'Вымышленный ответ [99]'}}]}))
    generated = answer('alpha', result, replace(cfg, api_url='https://api.example.org/v1', api_key='fixture-secret'))
    assert generated['answer_mode'] == 'extractive' and '[1]' in generated['answer']
    assert 'fixture-secret' not in str(generated)
    assert not valid_citations('Факт [1].\n\nДругой факт без ссылки.', 2)
    assert valid_citations('Недостаточно данных в источниках.', 0)


def test_public_urls_and_article_container_priority(monkeypatch):
    assert canonical_url('https://Example.org/a?utm_source=x&x=1#part') == 'https://example.org/a?x=1'
    for url in ('http://127.0.0.1', 'http://[::1]', 'http://169.254.169.254', 'file:///etc/passwd', 'https://user:pass@example.org'):
        with pytest.raises(FetchError):
            validate_public_url(url)
    body = 'Полезный текст статьи и подробные результаты исследования. ' * 5
    article = parse_article(f'<title>Title</title><article>MENU<div id="post-content-body">{body}</div>COMMENTS</article>', 'https://example.org/a')
    assert 'MENU' not in article['text'] and 'COMMENTS' not in article['text']


def test_jobs_are_persisted_and_claimed_once(fixture):
    db, _, _, source = fixture
    ident = db.enqueue('ingest', source['id'])
    assert db.enqueue('ingest', source['id']) == ident
    assert db.claim('a')['id'] == ident
    assert Database(db.path).claim('b') is None


def test_nplus1_body_excludes_next_preview_and_reads_jsonld_date():
    body = 'Основной текст исследования о гориллах и их взаимодействии. ' * 4
    html = ('<title>Гориллы</title><script type="application/ld+json">'
            '{"@type":"NewsArticle","datePublished":"2026-10-30T17:45:00+03:00"}</script>'
            '<div data-heading-wrap><div data-lmb-0>' + body +
            '<div data-lmb-0>Сноска относится к этой статье.</div></div></div>'
            '<aside><div data-lmb-0>Анонс следующей статьи: орангутаны и смех.</div></aside>')
    doc = parse_article(html, 'https://nplus1.ru/news/2026/10/30/example')
    assert 'орангутаны' not in doc['text'] and 'Сноска' in doc['text']
    assert doc['published_at'] == '2026-10-30T14:45:00+00:00'


def test_benchmark_rejects_a_different_corpus(fixture, tmp_path):
    from siterag.evaluation import evaluate
    db, models, cfg, _ = fixture
    with pytest.raises(ValueError, match='Корпус отличается'):
        evaluate(SearchEngine(db, models, cfg), 'benchmarks/questions.jsonl', tmp_path / 'results.json')


def test_nplus1_layout_change_does_not_import_the_entire_page():
    with pytest.raises(FetchError, match='границы основного текста'):
        parse_article('<title>Article</title><p>' + 'Text and related article previews. ' * 20 + '</p>', 'https://nplus1.ru/news/example')


def test_default_source_migration_preserves_documents_and_user_choices(tmp_path):
    db = Database(tmp_path / 'rag.sqlite3')
    old = db.add_source('N+1 · наука', 'rss', 'https://nplus1.ru/rss')
    custom = db.add_source('Мой источник', 'rss', 'https://example.org/rss')
    db.upsert(old['id'], 'Архивная статья', 'https://nplus1.ru/news/archive', 'Сохранённый текст')
    db.seed()
    replacement = next(s for s in db.sources() if s['url'] == 'https://elementy.ru/rss/news')
    assert not db.source(old['id'])['enabled']
    assert db.source(custom['id'])['enabled']
    with db.connect() as connection:
        assert connection.execute('SELECT COUNT(*) FROM documents').fetchone()[0] == 1
    assert not db.documents()  # The archive remains stored but is excluded from search.
    db.enable(replacement['id'], False)
    db.enable(old['id'], True)  # A later explicit administrator choice must survive restarts.
    db.seed()
    assert not db.source(replacement['id'])['enabled']
    assert db.source(old['id'])['enabled']
    assert len(db.sources()) == 4


def test_elementy_body_excludes_comments_and_recommendations():
    body = 'Основной текст исследования: результаты и объяснение эксперимента. ' * 5
    html = ('<title>Открытие</title><nav>Меню</nav><div class="itemblock">'
            '<div class="memo"><p>' + body + '</p></div>'
            '<div class="forumctrls">Комментарии пользователей</div></div>'
            '<div class="other_block">Другие новости</div>')
    doc = parse_article(html, 'https://elementy.ru/novosti_nauki/example')
    assert doc['text'] == body.strip()
    with pytest.raises(FetchError, match='границы основного текста'):
        parse_article('<title>Ошибка</title><p>' + body + '</p>', doc['url'])


@pytest.mark.parametrize('fetched,skipped,expected', [(2, 0, 'done'), (2, 1, 'partial'), (0, 1, 'failed')])
def test_worker_reports_page_failures_and_keeps_saved_data(fixture, monkeypatch, fetched, skipped, expected):
    from siterag.worker import run_job
    db, models, cfg, source = fixture
    documents = db.documents()
    monkeypatch.setattr('siterag.worker.collect', lambda *args: {
        'fetched': fetched, 'changed': 0, 'skipped': skipped,
        'errors': [{'url': 'https://example.org/timeout', 'error': 'Таймаут'}] if skipped else [],
    })
    ident = db.enqueue('ingest', source['id'])
    run_job(db, models, cfg, db.claim('test'))
    job = next(j for j in db.jobs() if j['id'] == ident)
    assert job['state'] == expected
    assert bool(job['error']) == bool(skipped)
    assert bool(db.source(source['id'])['last_error']) == bool(skipped)
    assert db.documents() == documents
    assert SearchEngine(db, models, cfg).search('alpha', rerank='never')['hits']


def test_unchanged_ingest_reuses_index_but_explicit_rebuild_does_not(fixture, monkeypatch):
    from siterag.worker import run_job
    db, models, cfg, source = fixture
    before = SearchEngine(db, models, cfg).search('alpha', rerank='never')['meta']['index_version']
    monkeypatch.setattr('siterag.worker.collect', lambda *args: {'fetched': 3, 'changed': 0, 'skipped': 0, 'errors': []})
    db.enqueue('ingest', source['id'])
    run_job(db, models, cfg, db.claim('test'))
    assert db.jobs()[0]['result']['index_reused']
    assert db.jobs()[0]['result']['index']['version'] == before
    db.enqueue('index')
    run_job(db, models, cfg, db.claim('test'))
    assert not db.jobs()[0]['result']['index_reused']
    assert db.jobs()[0]['result']['index']['version'] != before


def test_ingest_rebuilds_a_corrupt_index(fixture, monkeypatch):
    from siterag.worker import run_job
    db, models, cfg, source = fixture
    version = SearchEngine(db, models, cfg).search('alpha', rerank='never')['meta']['index_version']
    (cfg.data_dir / 'indices' / version / 'chunks.json').write_text('[]')
    monkeypatch.setattr('siterag.worker.collect', lambda *args: {'fetched': 3, 'changed': 0, 'skipped': 0, 'errors': []})
    db.enqueue('ingest', source['id'])
    run_job(db, models, cfg, db.claim('test'))
    assert db.jobs()[0]['state'] == 'done' and not db.jobs()[0]['result']['index_reused']
    assert SearchEngine(db, models, cfg).search('alpha', rerank='never')['hits']
