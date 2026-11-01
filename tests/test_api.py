from dataclasses import replace

from fastapi.testclient import TestClient

from siterag.api import create_app
from siterag.config import Settings


def client(tmp_path, **changes):
    cfg = Settings(data_dir=tmp_path, models_enabled=False, admin_password='a-test-password-12345', session_secret='x' * 40)
    return TestClient(create_app(replace(cfg, **changes)))


def test_public_endpoints_and_validation(tmp_path):
    app = client(tmp_path)
    assert app.get('/health').status_code == 200
    assert app.get('/sources').json()
    assert app.get('/search', params={'q': 'question'}).json()['hits'] == []
    assert app.get('/ask', params={'q': 'question'}).json()['answer'] == 'Недостаточно данных в источниках.'
    assert app.post('/ask', json={'q': 'question', 'use_llm': False}).status_code == 200
    assert app.get('/search', params={'q': ' ', 'limit': 0}).status_code == 422
    assert app.get('/search', params={'q': 'q', 'date_from': '2026-11-02', 'date_to': '2026-11-01'}).status_code == 422


def test_admin_session_csrf_upload_and_jobs(tmp_path):
    app = client(tmp_path)
    assert app.get('/admin/status').status_code == 401
    assert app.post('/admin/login', json={'password': 'wrong'}).status_code == 401
    login = app.post('/admin/login', json={'password': 'a-test-password-12345'})
    assert login.status_code == 200 and 'HttpOnly' in login.headers['set-cookie']
    csrf = login.json()['csrf']
    assert app.get('/admin/session').json()['csrf'] == csrf
    assert app.post('/admin/jobs', json={}).status_code == 403
    headers = {'X-CSRF-Token': csrf}
    uploaded = app.post('/admin/upload', headers=headers, files={'file': ('test.md', 'Настоящий загруженный документ. Текст для проверки сохранения и поиска.'.encode())})
    assert uploaded.status_code == 202
    assert app.get(uploaded.json()['url']).status_code == 200
    assert app.get('/admin/jobs').json()[0]['state'] == 'queued'
    assert app.post('/admin/upload', headers=headers, files={'file': ('bad.exe', b'bad')}).status_code == 422
    assert app.post('/admin/logout', headers=headers).status_code == 200
    assert app.get('/admin/status').status_code == 401


def test_get_and_post_ask_share_rate_limit(tmp_path):
    app = client(tmp_path, ask_rate=1)
    assert app.get('/ask', params={'q': 'hello'}).status_code == 200
    response = app.post('/ask', json={'q': 'hello'})
    assert response.status_code == 429 and response.headers['retry-after'] == '60'


def test_password_configuration_rejects_weak_secret(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        create_app(Settings(data_dir=tmp_path, admin_password='short'))
