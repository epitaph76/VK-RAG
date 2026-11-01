import calendar
import json
import ipaddress
import re
import socket
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import feedparser
import requests
from bs4 import BeautifulSoup


class FetchError(ValueError):
    pass


def canonical_url(url):
    parts = urlsplit(url.strip())
    if parts.scheme not in ('http', 'https') or not parts.hostname or parts.username or parts.password:
        raise FetchError('Нужен публичный HTTP(S) URL без пароля')
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not k.lower().startswith('utm_') and k.lower() not in ('fbclid', 'gclid')]
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path or '/', urlencode(query), ''))


def validate_public_url(url):
    url = canonical_url(url)
    parts = urlsplit(url)
    if parts.port not in (None, 80, 443):
        raise FetchError('Разрешены только порты HTTP/HTTPS')
    try:
        addresses = socket.getaddrinfo(parts.hostname, parts.port or (443 if parts.scheme == 'https' else 80), type=socket.SOCK_STREAM)
    except OSError:
        raise FetchError('Не удалось определить адрес источника') from None
    if not addresses or any(not ipaddress.ip_address(addr[4][0]).is_global for addr in addresses):
        raise FetchError('Локальные и служебные адреса источников запрещены')
    return url


RETRY_STATUSES = {408, 429, 500, 502, 503, 504}
FETCH_ATTEMPTS = 3


def retry_delay(value, attempt):
    delay = 2 ** (attempt - 1)
    if value:
        try:
            requested = float(value)
        except ValueError:
            try:
                requested = (parsedate_to_datetime(value) - datetime.now(timezone.utc)).total_seconds()
            except (TypeError, ValueError, OverflowError):
                requested = 0
        if requested > 30:
            raise FetchError('Источник просит повторить запрос позже; автоматическое ожидание ограничено 30 с')
        delay = max(delay, requested) if requested > 0 else delay
    return delay


def fetch(url, *, session=None, on_retry=None):
    """Bounded GET retries; every redirect/retry is validated, every body is capped."""
    if session is None:
        with requests.Session() as client:
            return fetch(url, session=client, on_retry=on_retry)
    for _ in range(5):
        for attempt in range(1, FETCH_ATTEMPTS + 1):
            url = validate_public_url(url)
            wait_header = None
            try:
                response = session.get(url, headers={'User-Agent': requests.utils.default_user_agent()},
                                       allow_redirects=False, timeout=(10, 25), stream=True)
                with response:
                    if response.is_redirect:
                        url = urljoin(url, response.headers.get('Location', ''))
                        break
                    if response.status_code in RETRY_STATUSES:
                        reason = f'Источник вернул HTTP {response.status_code}'
                        wait_header = response.headers.get('Retry-After')
                    elif response.status_code != 200:
                        raise FetchError(f'Источник вернул HTTP {response.status_code}; повтор не выполняется')
                    else:
                        payload = bytearray()
                        for block in response.iter_content(65536):
                            payload.extend(block)
                            if len(payload) > 8 * 1024 * 1024:
                                raise FetchError('Размер страницы превышает 8 МБ')
                        encoding = response.encoding
                        if not encoding or encoding.lower() == 'iso-8859-1':
                            encoding = 'utf-8'
                        return bytes(payload).decode(encoding, errors='replace'), url
            except requests.exceptions.SSLError:
                raise FetchError('Ошибка TLS; проверка сертификата не отключается') from None
            except requests.Timeout:
                reason = 'Сетевой таймаут (соединение: 10 с, ожидание данных: 25 с)'
            except (requests.ConnectionError, requests.exceptions.ChunkedEncodingError):
                reason = 'Соединение с источником прервано или не установлено'
            except requests.RequestException:
                raise FetchError('Ошибка HTTP-запроса к источнику') from None
            if attempt == FETCH_ATTEMPTS:
                raise FetchError(f'{reason}; выполнено попыток: {attempt}')
            delay = retry_delay(wait_header, attempt)
            if on_retry:
                on_retry({'url': url, 'attempt': attempt + 1, 'max_attempts': FETCH_ATTEMPTS,
                          'reason': reason, 'delay_seconds': delay})
            time.sleep(delay)
    raise FetchError('Слишком много перенаправлений')


def normalize_date(value):
    if not value:
        return None
    try:
        date = datetime.fromisoformat(value.replace('Z', '+00:00'))
    except (ValueError, TypeError):
        try:
            date = parsedate_to_datetime(value)
        except (ValueError, TypeError):
            return None
    # A date without an offset does not establish a publication timezone.
    if date.tzinfo is None:
        return None
    return date.astimezone(timezone.utc).isoformat()


def parse_article(html, url):
    import trafilatura
    soup = BeautifulSoup(html, 'html.parser')
    title_tag = soup.find('meta', property='og:title')
    title = (title_tag.get('content', '') if title_tag else '') or (soup.title.get_text(' ', strip=True) if soup.title else url)
    date_tag = soup.find('meta', property='article:published_time') or soup.find('meta', attrs={'name': 'date'})
    time_tag = soup.find('time', datetime=True)
    published = normalize_date(date_tag.get('content') if date_tag else time_tag.get('datetime') if time_tag else None)
    if not published:
        for script in soup.find_all('script', type='application/ld+json'):
            try:
                payload = json.loads(script.string or script.get_text())
                nodes = payload if isinstance(payload, list) else payload.get('@graph', [payload])
                for node in nodes:
                    if isinstance(node, dict) and node.get('@type') in ('NewsArticle', 'Article'):
                        published = normalize_date(node.get('datePublished'))
                        if published:
                            break
            except (ValueError, TypeError, AttributeError):
                continue
            if published:
                break
    # Source-specific article containers avoid comments, menus, and recommendations.
    selectors = ('#post-content-body', '.tm-article-body', '.topic-body', 'article')
    hostname = urlsplit(url).hostname
    if hostname in ('elementy.ru', 'www.elementy.ru'):
        container = soup.select_one('.itemblock > .memo')
        if container is None:
            raise FetchError('Не найдены границы основного текста «Элементов»; страница пропущена')
    elif hostname in ('nplus1.ru', 'www.nplus1.ru', 'nplus-1.ru'):
        container = soup.select_one('[data-heading-wrap] [data-lmb-0]')
        if container is None:
            raise FetchError('Не найдены границы основного текста N+1; страница пропущена')
    else:
        container = next((node for selector in selectors
                          if (node := soup.select_one(selector)) is not None), None)
    if container:
        for node in container.select('script,style,nav,footer,form,aside'):
            node.decompose()
        text = container.get_text('\n', strip=True)
    else:
        text = trafilatura.extract(html, url=url, include_comments=False, include_tables=True) or ''
    text = re.sub(r'\n{3,}', '\n\n', text).strip()
    if len(text) < 100:
        raise FetchError('На странице недостаточно текста статьи')
    return {'title': title.strip(), 'url': canonical_url(url), 'text': text, 'published_at': published}


def collect(database, source, limit=100, progress=None):
    # Reuse a source's verified TLS/TCP connections instead of reconnecting per article.
    with requests.Session() as session:
        return _collect(database, source, limit, progress, session)


def _collect(database, source, limit, progress, session):
    result = {'fetched': 0, 'changed': 0, 'skipped': 0, 'errors': [], 'retries': 0, 'retry_events': []}
    done, total = 0, 0
    def retry(event):
        result['retries'] += 1
        result['retry_events'].append(event)
        if progress:
            progress(done, total, result)
    if progress:
        progress(0, 0, result)
    try:
        html, final = fetch(source['url'], session=session, on_retry=retry)
    except FetchError as error:
        stage = 'RSS' if source['kind'] == 'rss' else 'Страница'
        raise FetchError(f'{stage}: {error}') from None
    entries = []
    if source['kind'] == 'rss':
        feed = feedparser.parse(html)
        if not feed.entries:
            raise FetchError('RSS не содержит публикаций')
        seen = set()
        for entry in feed.entries:
            url = entry.get('link', '')
            if not url:
                continue
            url = canonical_url(urljoin(final, url))
            if urlsplit(url).hostname != urlsplit(final).hostname or url in seen:
                continue
            seen.add(url)
            entries.append((url, entry.get('published') or entry.get('updated')))
        entries = entries[:limit]
    else:
        entries = [(final, None)]
    total = len(entries)
    for number, (url, rss_date) in enumerate(entries, 1):
        try:
            body, actual = (html, final) if source['kind'] == 'html' else fetch(url, session=session, on_retry=retry)
            doc = parse_article(body, actual)
            doc['published_at'] = doc['published_at'] or normalize_date(rss_date)
            result['changed'] += database.upsert(source['id'], **doc)
            result['fetched'] += 1
        except FetchError as error:
            result['skipped'] += 1
            result['errors'].append({'url': url, 'error': str(error)})
        done = number
        if progress:
            progress(done, total, result)
        if source['kind'] == 'rss':
            time.sleep(0.25)
    return result
