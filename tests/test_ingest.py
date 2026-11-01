import socket

import pytest
import requests

from siterag.db import Database
from siterag.ingest import FetchError, collect, fetch


class Response:
    def __init__(self, status=200, body=b'article', headers=None, blocks=None):
        self.status_code = status
        self.headers = headers or {}
        self.is_redirect = status in (301, 302, 303, 307, 308)
        self.encoding = 'utf-8'
        self.blocks = blocks if blocks is not None else [body]
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def iter_content(self, size):
        for block in self.blocks:
            if isinstance(block, Exception):
                raise block
            yield block


class Session:
    def __init__(self, outcomes):
        self.outcomes = iter(outcomes)
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def get(self, url, **kwargs):
        assert kwargs['allow_redirects'] is False
        assert kwargs.get('verify', True) is True
        self.calls.append(url)
        outcome = next(self.outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.fixture(autouse=True)
def public_dns_and_no_sleep(monkeypatch):
    monkeypatch.setattr(socket, 'getaddrinfo', lambda host, port, **kwargs: [
        (socket.AF_INET, socket.SOCK_STREAM, 6, '',
         ('127.0.0.1' if host == '127.0.0.1' else '8.8.8.8', port))])
    delays = []
    monkeypatch.setattr('siterag.ingest.time.sleep', delays.append)
    return delays


def test_transient_timeouts_recover_with_bounded_backoff(public_dns_and_no_sleep):
    session = Session([requests.ConnectTimeout(), requests.ReadTimeout(), Response(body=b'ok')])
    events = []
    assert fetch('https://example.org/rss', session=session, on_retry=events.append)[0] == 'ok'
    assert len(session.calls) == 3
    assert public_dns_and_no_sleep == [1, 2]
    assert [e['attempt'] for e in events] == [2, 3]


def test_persistent_timeout_stops_and_reports_attempt_count(public_dns_and_no_sleep):
    session = Session([requests.ReadTimeout()] * 3)
    with pytest.raises(FetchError, match='Сетевой таймаут.*попыток: 3'):
        fetch('https://example.org/rss', session=session)
    assert len(session.calls) == 3
    assert public_dns_and_no_sleep == [1, 2]


def test_retryable_http_respects_retry_after_and_closes_response(public_dns_and_no_sleep):
    temporary = Response(503, headers={'Retry-After': '4'})
    session = Session([temporary, Response(body=b'ok')])
    assert fetch('https://example.org/rss', session=session)[0] == 'ok'
    assert temporary.closed
    assert public_dns_and_no_sleep == [4]


@pytest.mark.parametrize('outcome,expected', [
    (Response(403), 'HTTP 403'),
    (requests.exceptions.SSLError('certificate verification failed'), 'TLS'),
    (Response(429, headers={'Retry-After': '120'}), 'повторить запрос позже'),
])
def test_permanent_errors_do_not_retry_or_weaken_tls(outcome, expected, public_dns_and_no_sleep):
    session = Session([outcome])
    with pytest.raises(FetchError, match=expected):
        fetch('https://example.org/rss', session=session)
    assert len(session.calls) == 1
    assert not public_dns_and_no_sleep


def test_stream_disconnect_restarts_body_without_concatenating_partial_bytes():
    broken = Response(blocks=[b'partial', requests.exceptions.ChunkedEncodingError()])
    session = Session([broken, Response(body=b'complete')])
    assert fetch('https://example.org/article', session=session)[0] == 'complete'
    assert broken.closed


def test_redirect_to_private_address_is_rejected_before_request():
    session = Session([Response(302, headers={'Location': 'http://127.0.0.1/secret'})])
    with pytest.raises(FetchError, match='Локальные'):
        fetch('https://example.org/rss', session=session)
    assert session.calls == ['https://example.org/rss']


def test_collect_reuses_session_and_reports_rss_retry(tmp_path, monkeypatch):
    feed = b'<rss><channel><item><link>https://example.org/a</link></item><item><link>https://example.org/b</link></item></channel></rss>'
    article = ('<title>Article</title><article>' + 'Useful scientific article. ' * 20 + '</article>').encode()
    session = Session([requests.ConnectTimeout(), Response(body=feed), Response(body=article), Response(body=article)])
    monkeypatch.setattr('siterag.ingest.requests.Session', lambda: session)
    db = Database(tmp_path / 'rag.sqlite3')
    source = db.add_source('Example', 'rss', 'https://example.org/rss')
    progress = []
    report = collect(db, source, progress=lambda done, total, report: progress.append((done, total, report['retries'])))
    assert report['fetched'] == report['changed'] == 2
    assert report['retries'] == 1 and report['skipped'] == 0
    assert report['retry_events'][0]['url'] == source['url']
    assert (0, 0, 1) in progress and progress[-1] == (2, 2, 1)
    assert len(db.documents()) == 2 and len(session.calls) == 4
