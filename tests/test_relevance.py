from dataclasses import replace
from types import SimpleNamespace

import pytest

from siterag.config import Settings
from siterag.generation import REFUSAL, answer
from siterag.relevance import expand_query, select_context


def hit(ident, text, score=None, dense=None):
    return {'id': ident, 'text': text, 'title': ident, 'url': 'https://example.org/' + ident,
            'rerank_score': score, 'dense_score': dense, 'retrieval_score': .03}


def result(hits):
    return {'hits': hits, 'meta': {'warnings': [], 'timings_ms': {'total': 1}}}


def test_context_drops_negative_candidates_without_filling_top_five():
    candidates = [hit('ml', 'Новая модель машинного обучения.', 2),
                  hit('animals', 'Социальное обучение животных.', -3),
                  hit('medicine', 'Оптогенетика и нейроны.', -2),
                  hit('paper', 'Результаты разработки модели.', .2)]
    chosen, meta = select_context('какие последние новости в ml', result(candidates), Settings())
    assert [h['id'] for h in chosen] == ['ml', 'paper']
    assert meta['selected_count'] == 2 and len(meta['rejected']) == 2


def test_no_relevant_evidence_abstains_without_calling_llm(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('An unsupported question must not be sent to the LLM')
    monkeypatch.setattr('siterag.generation.requests.post', unexpected)
    original = result([hit('animals', 'Социальное обучение животных.', -3)])
    cfg = Settings(api_url='https://example.org/v1', api_key='test-key')
    response = answer('какие последние новости в ml', original, cfg)
    assert response['answer'] == REFUSAL and response['sources'] == []
    assert response['meta']['context_filter']['selected_count'] == 0
    assert original['meta']['warnings'] == []


def test_never_mode_and_reranker_outage_require_topic_support():
    candidates = [hit('animals', 'Обучение животных и традиции шимпанзе.', dense=.9),
                  hit('ml', 'Исследователи улучшили методы машинного обучения.', dense=.85),
                  hit('medicine', 'Ученые исследовали нейроны и свет.', dense=.92)]
    chosen, meta = select_context('какие последние новости в ml', result(candidates), Settings())
    assert [h['id'] for h in chosen] == ['ml']
    assert meta['methods'] == ['lexical_dense_fallback']


@pytest.mark.parametrize('dense,expected', [(None, True), (.9, True), (.7, False), (float('nan'), False)])
def test_fallback_keeps_bm25_without_dense_but_rejects_weak_dense(dense, expected):
    chosen, _ = select_context('квантовые вычисления', result([
        hit('a', 'Квантовым вычислениям посвящено новое исследование.', dense=dense)]), Settings())
    assert bool(chosen) is expected


def test_threshold_is_a_configurable_logit_not_rrf_or_probability():
    candidate = hit('a', 'irrelevant', score=-.2)
    assert not select_context('question', result([candidate]), Settings())[0]
    assert select_context('question', result([candidate]), replace(Settings(), context_min_rerank_score=-1))[0]


def test_aliases_respect_word_boundaries():
    expanded = expand_query('какие последние новости в ML')
    assert 'машинное обучение' in expanded
    assert expand_query('HTML и семейство') == 'HTML и семейство'


def test_only_cited_sources_are_returned_and_references_are_renumbered(monkeypatch):
    monkeypatch.setattr('siterag.generation.requests.post', lambda *args, **kwargs:
        SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            'choices': [{'message': {'content': 'Результат эксперимента [2].'}}]}))
    cfg = Settings(api_url='https://example.org/v1', api_key='test-key')
    response = answer('question', result([hit('a', 'Text A', 1), hit('b', 'Text B', 1)]), cfg)
    assert response['answer_mode'] == 'llm'
    assert response['answer'] == 'Результат эксперимента [1].'
    assert [h['id'] for h in response['sources']] == ['b']


def test_llm_refusal_does_not_leave_source_cards(monkeypatch):
    monkeypatch.setattr('siterag.generation.requests.post', lambda *args, **kwargs:
        SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            'choices': [{'message': {'content': REFUSAL}}]}))
    response = answer('question', result([hit('a', 'Potential evidence.', 1)]),
                      Settings(api_url='https://example.org/v1', api_key='test-key'))
    assert response['answer'] == REFUSAL and response['sources'] == []
