"""Context admission is separate from ranking; scores are uncalibrated signals."""
import math
import re
import threading
from functools import lru_cache

import snowballstemmer


ALIASES = {'ml': 'машинное обучение', 'ai': 'искусственный интеллект',
           'ии': 'искусственный интеллект', 'llm': 'большая языковая модель'}
ALIAS_PATTERN = re.compile(r'\b(ml|ai|ии|llm)\b', re.IGNORECASE)
STOPWORDS = set(('а без был была были быть в во вы где да для до его ее если есть еще же за зачем '
                 'здесь и из или их к как какой какая какие какого какие кто ли мне мы на над не но '
                 'о об от по под почему при про с со так такой такие там то тот у уже что это этот '
                 'эти я какие последние последний новый новые новости новость свежий расскажи покажи '
                 'найди пожалуйста информация материал материалы the a an and are as at be by for '
                 'from how in is it latest me new news of on or recent tell that this to was what '
                 'when where which who why with').split())
_local = threading.local()


def expand_query(query):
    return ALIAS_PATTERN.sub(lambda match: match.group(0) + ' ' + ALIASES[match.group(0).casefold()], query)


@lru_cache(maxsize=50000)
def stem(word):
    if not hasattr(_local, 'russian'):
        _local.russian = snowballstemmer.stemmer('russian')
        _local.english = snowballstemmer.stemmer('english')
    return (_local.russian if re.search('[а-я]', word) else _local.english).stemWord(word)


def terms(text):
    return {stem(word) for word in re.findall(r'\w+', expand_query(text.casefold().replace('ё', 'е')))
            if len(word) >= 2 and not word.isdecimal() and word not in STOPWORDS}


def select_context(question, search_result, settings):
    candidates = search_result['hits']
    wanted = terms(question)
    accepted, rejected = [], []
    methods = set()
    for hit in candidates:
        score = hit.get('rerank_score')
        reason = None
        if score is not None:
            methods.add('reranker_logit')
            if not math.isfinite(score) or score < settings.context_min_rerank_score:
                reason = 'reranker_below_threshold'
        else:
            # No additional reranker is run in never / confident-keep / outage modes.
            methods.add('lexical_dense_fallback')
            overlap = len(wanted.intersection(terms(hit['text'])))
            coverage = overlap / len(wanted) if wanted else 0
            dense_score = hit.get('dense_score')
            if not wanted or overlap < min(2, len(wanted)) or coverage < settings.context_min_term_coverage:
                reason = 'insufficient_query_term_support'
            elif dense_score is not None and (not math.isfinite(dense_score) or dense_score < settings.context_min_dense_score):
                reason = 'dense_below_threshold'
        if reason:
            rejected.append({'id': hit['id'], 'reason': reason})
        else:
            accepted.append(hit)
    return accepted[:5], {
        'policy_version': 'context-v1', 'methods': sorted(methods), 'input_count': len(candidates),
        'accepted_count': len(accepted), 'selected_count': min(5, len(accepted)), 'rejected': rejected,
        'thresholds': {'reranker_logit': settings.context_min_rerank_score,
                       'dense_cosine': settings.context_min_dense_score,
                       'term_coverage': settings.context_min_term_coverage},
    }
