from dataclasses import replace
from types import SimpleNamespace
import json

import pytest
import requests

from siterag.config import Settings
from siterag.generation import REFUSAL, answer, extractive
from siterag.relevance import query_plan, select_context


def hit(ident, text, score=None, dense=None):
    return {'id': ident, 'text': text, 'title': ident, 'url': 'https://example.org/' + ident,
            'rerank_score': score, 'dense_score': dense, 'retrieval_score': .03}


def result(hits):
    return {'hits': hits, 'meta': {'warnings': [], 'timings_ms': {'total': 1}}}


def test_context_drops_negative_candidates_without_filling_top_five():
    candidates = [hit('ml', 'Новая модель машинного обучения.', 2),
                  hit('animals', 'Социальное обучение животных.', -3),
                  hit('medicine', 'Оптогенетика и нейроны.', -2),
                  hit('paper', 'Результаты разработки большой языковой модели.', .2)]
    chosen, meta = select_context('какие последние новости в ml', result(candidates), Settings())
    assert [h['id'] for h in chosen] == ['ml', 'paper']
    assert meta['selected_count'] == 2 and len(meta['rejected']) == 2


def test_no_relevant_evidence_abstains_when_llm_is_disabled(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('An unsupported question must not be sent to the LLM')
    monkeypatch.setattr('siterag.generation.requests.post', unexpected)
    original = result([hit('animals', 'Социальное обучение животных.', -3)])
    cfg = Settings(api_url='https://example.org/v1', api_key='test-key')
    response = answer('какие последние новости в ml', original, cfg, use_llm=False)
    assert response['answer'] == REFUSAL and response['sources'] == []
    assert response['meta']['context_filter']['selected_count'] == 0
    assert original['meta']['warnings'] == []
    assert response['meta']['generation'] == {'requested': False, 'attempted': False,
        'status': 'disabled', 'fallback_reason': 'user_disabled'}


def test_never_mode_and_reranker_outage_require_topic_support():
    candidates = [hit('animals', 'Обучение животных и традиции шимпанзе.', dense=.9),
                  hit('ml', 'Исследователи улучшили методы машинного обучения.', dense=.85),
                  hit('medicine', 'Ученые исследовали нейроны и свет.', dense=.92)]
    chosen, meta = select_context('какие последние новости в ml', result(candidates), Settings())
    assert [h['id'] for h in chosen] == ['ml']
    assert meta['methods'] == ['explicit_topic_support']


@pytest.mark.parametrize('dense,expected', [(None, True), (.9, True), (.7, False), (float('nan'), False)])
def test_fallback_keeps_bm25_without_dense_but_rejects_weak_dense(dense, expected):
    chosen, _ = select_context('квантовые вычисления алгоритмы', result([
        hit('a', 'Квантовым вычислениям посвящено новое исследование.', dense=dense)]), Settings())
    assert bool(chosen) is expected


def test_threshold_is_a_configurable_logit_not_rrf_or_probability():
    candidate = hit('a', 'irrelevant', score=-.2)
    assert not select_context('question', result([candidate]), Settings())[0]
    assert select_context('question', result([candidate]), replace(Settings(), context_min_rerank_score=-1))[0]


def test_aliases_respect_word_boundaries():
    expanded = query_plan('какие последние новости в ML')['topic']
    assert 'машинное обучение' in expanded
    assert query_plan('HTML и семейство')['topic'] == 'HTML и семейство'


def test_only_cited_sources_are_returned_and_references_are_renumbered(monkeypatch):
    monkeypatch.setattr('siterag.generation.requests.post', lambda *args, **kwargs:
        SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            'choices': [{'message': {'content': json.dumps({'status': 'answered', 'relevant_sources': [2],
                'answer': 'Результат эксперимента [2].'})}}]}))
    cfg = Settings(api_url='https://example.org/v1', api_key='test-key')
    response = answer('question', result([hit('a', 'Text A', 1), hit('b', 'Text B', 1)]), cfg)
    assert response['answer_mode'] == 'llm'
    assert response['answer'] == 'Результат эксперимента [1].'
    assert [h['id'] for h in response['sources']] == ['b']
    assert response['meta']['generation'] == {'requested': True, 'attempted': True,
        'status': 'generated', 'fallback_reason': None}


def test_llm_refusal_does_not_leave_source_cards(monkeypatch):
    monkeypatch.setattr('siterag.generation.requests.post', lambda *args, **kwargs:
        SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            'choices': [{'message': {'content': REFUSAL}}]}))
    response = answer('question', result([hit('a', 'Potential evidence.', 1)]),
                      Settings(api_url='https://example.org/v1', api_key='test-key'))
    assert response['answer'] == REFUSAL and response['sources'] == []


def test_negative_logit_cannot_veto_explicit_news_topic():
    candidates = [hit('ml', 'Новый тренд машинного обучения — персональные AI-ассистенты.', -2),
                  hit('animals', 'Социальное обучение животных.', -1)]
    chosen, meta = select_context('какие последние новости в ml', result(candidates), Settings())
    assert [h['id'] for h in chosen] == ['ml']
    assert meta['rescued_by_explicit_topic'] == ['ml']


def test_decision_models_require_decisions_and_models_not_just_learning():
    candidates = [hit('risk', 'Исследователи сравнили модели принятия решений и их склонность к риску.', -3.5),
                  hit('animals', 'Модели социального обучения животных.', -1),
                  hit('fatigue', 'Принятие решений вызывает усталость человека.', -1)]
    chosen, _ = select_context('какие последние новости в decision моделях', result(candidates), Settings())
    assert [h['id'] for h in chosen] == ['risk']


def test_news_wrapper_does_not_become_the_retrieval_topic():
    assert query_plan('какие последние новости в ml') == {'topic': 'машинное обучение', 'recent': True}
    assert query_plan('какие последние новости в decision моделях')['topic'] == 'модели принятия решений'
    assert query_plan('Почему древнейшего осьминога признали наутилусом?')['recent'] is False


def test_news_context_preserves_dates_and_document_diversity():
    candidates = [hit('old', 'Новые методы машинного обучения.', 1) | {'published_at': '2025-01-01'},
                  hit('new', 'Современные методы машинного обучения.', -2) | {'published_at': '2026-11-01'}]
    chosen, _ = select_context('какие последние новости в ml', result(candidates), Settings())
    assert [h['id'] for h in chosen] == ['new', 'old']


def test_semantic_selection_can_accept_paraphrase_despite_negative_logit(monkeypatch):
    monkeypatch.setattr('siterag.generation.requests.post', lambda *args, **kwargs:
        SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            'choices': [{'message': {'content': json.dumps({'status': 'answered', 'relevant_sources': [1],
                'answer': 'Исследование сравнивает рискованность выбора разных систем [1].'})}}]}))
    cfg = Settings(api_url='https://example.org/v1', api_key='test-key')
    response = answer('Как модели принимают решения?', result([
        hit('paper', 'Исследователи сравнили предпочтения риска у языковых систем.', -3)]), cfg)
    assert response['answer_mode'] == 'llm' and len(response['sources']) == 1
    assert response['meta']['context_filter']['methods'] == ['llm_semantic']


def test_empty_search_never_calls_llm(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError('No candidates')
    monkeypatch.setattr('siterag.generation.requests.post', unexpected)
    response = answer('question', result([]), Settings(api_url='https://example.org/v1', api_key='test-key'))
    assert response['answer'] == REFUSAL and not response['sources']


def test_semantic_selection_rejects_citations_outside_its_selection(monkeypatch):
    monkeypatch.setattr('siterag.generation.requests.post', lambda *args, **kwargs:
        SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            'choices': [{'message': {'content': json.dumps({'status': 'answered', 'relevant_sources': [1],
                'answer': 'Unsupported answer [2].'})}}]}))
    response = answer('question', result([hit('a', 'Unrelated A.', -3), hit('b', 'Unrelated B.', -3)]),
                      Settings(api_url='https://example.org/v1', api_key='test-key'))
    assert response['answer_mode'] == 'extractive' and not response['sources']


def test_extractive_news_uses_topic_evidence_instead_of_common_question_words():
    text = ('Температура в помещении и шум в офисе влияют на состояние мозга. '
            'Новый тренд машинного обучения — персональные ассистенты на основе LLM.')
    response = extractive('какие последние новости в ml', [hit('paper', text)])
    assert 'ассистенты' in response and 'Температура' not in response


@pytest.mark.parametrize('failure,expected', [
    (requests.Timeout(), 'timeout'),
    ('', 'empty_response'),
    ('{broken json', 'invalid_json'),
    (json.dumps({'status': 'answered', 'relevant_sources': [1], 'answer': 'Wrong citation [99].'}), 'invalid_citations'),
    (json.dumps({'status': 'answered', 'relevant_sources': [99], 'answer': 'Answer [1].'}), 'invalid_semantic_sources'),
    (requests.HTTPError(response=SimpleNamespace(status_code=502)), 'http_502'),
])
def test_automatic_fallback_preserves_requested_setting_and_reports_reason(monkeypatch, failure, expected):
    def post(*args, **kwargs):
        if isinstance(failure, Exception):
            raise failure
        return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {
            'choices': [{'message': {'content': failure}}]})
    monkeypatch.setattr('siterag.generation.requests.post', post)
    response = answer('DeepSeek', result([hit('paper', 'DeepSeek performed well in the experiment.', -2)]),
                      Settings(api_url='https://example.org/v1', api_key='test-key'))
    assert response['answer_mode'] == 'extractive' and response['sources']
    assert response['meta']['generation'] == {'requested': True, 'attempted': True,
        'status': 'fallback', 'fallback_reason': expected}
    assert response['meta']['llm_error'] == expected
    assert 'test-key' not in str(response)


def test_unconfigured_generation_is_different_from_user_disabled():
    response = answer('DeepSeek', result([hit('paper', 'DeepSeek experimental results.', -2)]),
                      Settings(api_url='', api_key=''))
    assert response['meta']['generation'] == {'requested': True, 'attempted': False,
        'status': 'fallback', 'fallback_reason': 'not_configured'}
