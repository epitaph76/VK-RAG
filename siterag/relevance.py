"""Context admission is separate from ranking; scores are uncalibrated signals."""
import math
import re
import threading
from functools import lru_cache

import snowballstemmer


ALIASES = {'ml': 'машинное обучение', 'ai': 'искусственный интеллект',
           'ии': 'искусственный интеллект', 'llm': 'большая языковая модель'}
ALIAS_PATTERN = re.compile(r'\b(ml|ai|ии|llm)\b', re.IGNORECASE)
CONCEPT_PATTERNS = {
    'ml': re.compile(r'\b(?:ml|machine\s+learning|машинн\w*\s+обучени\w*|deep\s+learning|глубок\w*\s+обучени\w*|искусственн\w*\s+нейронн\w*\s+сет\w*)\b', re.I),
    'llm': re.compile(r'\b(?:llms?|(?:больш\w*\s+)?языков\w+\s+модел\w*)\b', re.I),
    'ai': re.compile(r'\b(?:ai|ии|искусственн\w*\s+интеллект\w*)\b', re.I),
    'decision': re.compile(r'\b(?:decision(?:[ -]+making)?|приняти\w*\s+решени\w*)\b', re.I),
    'diffusion': re.compile(r'\b(?:diffusion|диффузионн\w*)\b', re.I),
}
NEWS_PATTERN = re.compile(
    r'^\s*(?:(?:какие|какая|что|расскажи|покажи|найди)\s+)?'
    r'(?:(?:есть|самые|самая|последние|последняя|свежие|свежая|новые|новая|актуальные)\s+)*'
    r'(?:новости|новость|нового)\s*(?:(?:в|во|по|про|о|об|из)\s+)?', re.I)
STOPWORDS = set(('а без был была были быть в во вы где да для до его ее если есть еще же за зачем '
                 'здесь и из или их к как какой какая какие какого какие кто ли мне мы на над не но '
                 'о об от по под почему при про с со так такой такие там то тот у уже что это этот '
                 'эти я какие последние последний новый новые новости новость свежий расскажи покажи '
                 'найди пожалуйста информация материал материалы the a an and are as at be by for '
                 'from how in is it latest me new news of on or recent tell that this to was what '
                 'when where which who why with').split())
_local = threading.local()


def query_plan(question):
    """Separate a news request's topic from its conversational framing."""
    match = NEWS_PATTERN.match(question)
    recent = bool(match and question[match.end():].strip(' ?.!'))
    topic = question[match.end():].strip(' ?.!') if recent else question
    topic = re.sub(r'\bdecision\s+(?:модел\w*|models?)\b', 'модели принятия решений', topic, flags=re.I)
    topic = re.sub(r'\bdecision\b', 'принятие решений', topic, flags=re.I)
    topic = ALIAS_PATTERN.sub(lambda m: ALIASES[m.group(0).casefold()], topic)
    return {'topic': topic, 'recent': recent}


@lru_cache(maxsize=50000)
def stem(word):
    if word in ('model', 'models'):
        return 'модел'
    if not hasattr(_local, 'russian'):
        _local.russian = snowballstemmer.stemmer('russian')
        _local.english = snowballstemmer.stemmer('english')
    return (_local.russian if re.search('[а-я]', word) else _local.english).stemWord(word)


def terms(text, *, evidence=False):
    text = text.casefold().replace('ё', 'е')
    concepts = {'@' + name for name, pattern in CONCEPT_PATTERNS.items() if pattern.search(text)}
    for pattern in CONCEPT_PATTERNS.values():
        text = pattern.sub(' ', text)
    if evidence:
        if '@llm' in concepts or '@diffusion' in concepts:
            concepts.add('@ml')
        if '@ml' in concepts:
            concepts.add('@ai')
    return concepts | {stem(word) for word in re.findall(r'\w+', text)
                       if len(word) >= 2 and not word.isdecimal() and word not in STOPWORDS}


def select_context(question, search_result, settings):
    candidates = search_result['hits']
    plan = query_plan(question)
    wanted = terms(plan['topic'])
    required_concepts = {term for term in wanted if term.startswith('@')}
    accepted, rejected, rescued = [], [], []
    methods = set()
    for hit in candidates:
        score = hit.get('rerank_score')
        reason = None
        evidence = terms(hit['text'], evidence=True)
        overlap = len(wanted.intersection(evidence))
        coverage = overlap / len(wanted) if wanted else 0
        concepts_supported = required_concepts <= evidence
        exact_support = bool(wanted) and concepts_supported and wanted <= evidence
        partial_support = (bool(wanted) and concepts_supported and
                           overlap >= min(2, len(wanted)) and coverage >= settings.context_min_term_coverage)
        dense_score = hit.get('dense_score')
        dense_supported = dense_score is None or (math.isfinite(dense_score) and dense_score >= settings.context_min_dense_score)
        if plan['recent'] and required_concepts and not concepts_supported:
            reason = 'requested_topic_not_supported'
        elif score is not None and math.isfinite(score) and score >= settings.context_min_rerank_score:
            methods.add('reranker_logit')
        elif exact_support:
            # An uncalibrated negative logit cannot veto explicit evidence for the topic.
            methods.add('explicit_topic_support')
            if score is not None:
                rescued.append(hit['id'])
        elif score is not None:
            methods.add('reranker_and_topic_support')
            reason = 'insufficient_topic_support'
        else:
            # No additional reranker is run in never / confident-keep / outage modes.
            methods.add('lexical_dense_fallback')
            if not partial_support:
                reason = 'insufficient_query_term_support'
            elif not dense_supported:
                reason = 'dense_below_threshold'
        if reason:
            rejected.append({'id': hit['id'], 'reason': reason})
        else:
            accepted.append(hit)
    if plan['recent']:
        accepted.sort(key=lambda hit: hit.get('published_at') or '', reverse=True)
    selected, per_document = [], {}
    for hit in accepted:
        document = hit.get('document_id', hit.get('url', hit['id']))
        if plan['recent'] and per_document.get(document, 0) >= 2:
            continue
        selected.append(hit)
        per_document[document] = per_document.get(document, 0) + 1
        if len(selected) == 5:
            break
    return selected, {
        'policy_version': 'context-v2', 'query_intent': plan, 'methods': sorted(methods), 'input_count': len(candidates),
        'accepted_count': len(accepted), 'selected_count': len(selected), 'rejected': rejected,
        'rescued_by_explicit_topic': rescued,
        'thresholds': {'reranker_logit': settings.context_min_rerank_score,
                       'dense_cosine': settings.context_min_dense_score,
                       'term_coverage': settings.context_min_term_coverage},
    }
