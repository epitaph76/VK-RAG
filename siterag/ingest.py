import calendar
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


def fetch(url):
    """Anonymous public downloads only; validate every redirect and cap decompressed bytes."""
    for _ in range(5):
        url = validate_public_url(url)
        try:
            response = requests.get(url, headers={'User-Agent': 'SiteRAG/1.0 (research; manual ingestion)'},
                                    allow_redirects=False, timeout=(10, 25), stream=True)
            with response:
                if response.is_redirect:
                    url = urljoin(url, response.headers.get('Location', ''))
                    continue
                if response.status_code != 200:
                    raise FetchError(f'Источник вернул HTTP {response.status_code}')
                payload = bytearray()
                for block in response.iter_content(65536):
                    payload.extend(block)
                    if len(payload) > 8 * 1024 * 1024:
                        raise FetchError('Размер страницы превышает 8 МБ')
                encoding = response.encoding
                if not encoding or encoding.lower() == 'iso-8859-1':
                    encoding = 'utf-8'
                return bytes(payload).decode(encoding, errors='replace'), url
        except requests.RequestException:
            raise FetchError('Источник недоступен или истекло время ожидания') from None
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
    # Source-specific article containers avoid comments, menus, and recommendations.
    container = next((node for selector in ('#post-content-body', '.tm-article-body', '.topic-body', 'article')
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
    html, final = fetch(source['url'])
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
    result = {'fetched': 0, 'changed': 0, 'skipped': 0, 'errors': []}
    for number, (url, rss_date) in enumerate(entries, 1):
        try:
            body, actual = (html, final) if source['kind'] == 'html' else fetch(url)
            doc = parse_article(body, actual)
            doc['published_at'] = doc['published_at'] or normalize_date(rss_date)
            result['changed'] += database.upsert(source['id'], **doc)
            result['fetched'] += 1
        except FetchError as error:
            result['skipped'] += 1
            result['errors'].append({'url': url, 'error': str(error)})
        if progress:
            progress(number, len(entries), result)
        if source['kind'] == 'rss':
            time.sleep(0.25)
    return result
