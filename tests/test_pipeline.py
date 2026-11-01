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
