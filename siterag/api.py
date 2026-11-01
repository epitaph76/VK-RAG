import hashlib
import hmac
import secrets
import sqlite3
import threading
import time
from collections import defaultdict, deque
from contextlib import contextmanager
from datetime import date
from typing import Literal

from fastapi import Depends, FastAPI, File, HTTPException, Query, Request, Response, UploadFile
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from pydantic import BaseModel, Field

from .config import Settings
from .db import Database
from .generation import answer
from .ingest import canonical_url, validate_public_url
from .models import Models
from .search import SearchEngine


class RateLimiter:
    def __init__(self):
        self.entries = defaultdict(deque)
        self.lock = threading.Lock()

    def check(self, key, limit):
        with self.lock:
            current = time.monotonic()
            if len(self.entries) >= 10000:
                for old in list(self.entries):
                    if not self.entries[old] or self.entries[old][-1] < current - 60:
                        del self.entries[old]
            if key not in self.entries and len(self.entries) >= 10000:
                raise HTTPException(429, 'Сервис занят. Повторите через минуту.', headers={'Retry-After': '60'})
            queue = self.entries[key]
            while queue and queue[0] <= current - 60:
                queue.popleft()
            if len(queue) >= limit:
                raise HTTPException(429, 'Слишком много запросов. Повторите через минуту.', headers={'Retry-After': '60'})
            queue.append(current)


class AskInput(BaseModel):
    q: str = Field(min_length=1, max_length=1000)
    limit: int = Field(default=10, ge=1, le=20)
    rerank: Literal['auto', 'always', 'never'] = 'auto'
    source_id: str | None = None
    date_from: date | None = None
    date_to: date | None = None
    use_llm: bool = True


class SourceInput(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    kind: Literal['rss', 'html']
    url: str = Field(min_length=1, max_length=2048)


class EnableInput(BaseModel):
    enabled: bool


class LoginInput(BaseModel):
    password: str = Field(min_length=1, max_length=512)


class JobInput(BaseModel):
    source_id: str | None = None
    kind: Literal['ingest', 'index'] = 'ingest'


def create_app(settings=None, models=None):
    cfg = settings or Settings()
    if cfg.admin_password and (len(cfg.admin_password) < 16 or len(cfg.session_secret) < 32):
        raise ValueError('ADMIN_PASSWORD: минимум 16 символов; SESSION_SECRET: минимум 32 символа')
    db = Database(cfg.database)
    db.seed()
    models = models or Models(cfg)
    engine = SearchEngine(db, models, cfg)
    limiter = RateLimiter()
    serializer = URLSafeTimedSerializer(cfg.session_secret or secrets.token_hex(32), salt='siterag-admin-v1-' + hashlib.sha256(cfg.admin_password.encode()).hexdigest())
    slots = threading.BoundedSemaphore(cfg.queue_size + 1)
    pipeline = threading.Lock()
    app = FastAPI(title='Срез · Site RAG', version='1.0.0', root_path=cfg.root_path)
    app.state.db, app.state.engine, app.state.models = db, engine, models

    @app.middleware('http')
    async def headers(request, call_next):
        response = await call_next(request)
        response.headers['Cache-Control'] = 'no-store'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Referrer-Policy'] = 'same-origin'
        return response

    def limit(request, group, count):
        # Uvicorn does not trust forwarded headers; only Nginx may set X-Real-IP.
        peer = request.client.host if request.client else 'unknown'
        from ipaddress import ip_address
        forwarded = request.headers.get('x-real-ip')
        try:
            internal = ip_address(peer).is_private
            if internal and forwarded:
                peer = str(ip_address(forwarded))
        except ValueError:
            pass
        limiter.check((group, peer), count)

    @contextmanager
    def pipeline_slot():
        if not slots.acquire(blocking=False):
            raise HTTPException(429, 'Очередь запросов заполнена', headers={'Retry-After': '15'})
        try:
            if not pipeline.acquire(timeout=90):
                raise HTTPException(429, 'Поиск занят. Попробуйте ещё раз.', headers={'Retry-After': '15'})
            try:
                yield
            finally:
                pipeline.release()
        finally:
            slots.release()

    def admin(request: Request):
        try:
            session = serializer.loads(request.cookies.get('rag_admin', ''), max_age=43200)
        except (BadSignature, SignatureExpired):
            raise HTTPException(401, 'Войдите как администратор') from None
        if not cfg.admin_password or session.get('role') != 'admin':
            raise HTTPException(401, 'Войдите как администратор')
        if request.method not in ('GET', 'HEAD') and not hmac.compare_digest(request.headers.get('x-csrf-token', ''), session['csrf']):
            raise HTTPException(403, 'Сессия устарела. Обновите страницу.')
        return session

    def run_search(args):
        try:
            return engine.search(**args)
        except ValueError as error:
            raise HTTPException(422, str(error)) from None
        except Exception:
            raise HTTPException(503, 'Индекс временно недоступен. Администратор может проверить базу.') from None

    @app.get('/health')
    def health():
        return {'status': 'ok', 'service': 'site-rag'}

    @app.get('/sources')
    def public_sources():
        return [{'id': s['id'], 'name': s['name'], 'kind': s['kind']} for s in db.sources(public=True)]

    @app.get('/search')
    def search(request: Request, q: str = Query(min_length=1, max_length=1000), limit_: int = Query(default=10, alias='limit', ge=1, le=20),
               rerank: Literal['auto', 'always', 'never'] = 'auto', source_id: str | None = None,
               date_from: date | None = None, date_to: date | None = None):
        limit(request, 'search', cfg.search_rate)
        with pipeline_slot():
            return run_search(dict(q=q, limit=limit_, rerank=rerank, source_id=source_id, date_from=date_from, date_to=date_to))

    def ask_common(request, payload):
        limit(request, 'ask', cfg.ask_rate)
        with pipeline_slot():
            values = payload.model_dump()
            use_llm = values.pop('use_llm')
            # The answer always has the best five even if the caller requests a smaller search listing.
            values['limit'] = 20  # Admission considers the full candidate pool, then selects up to five.
            return answer(payload.q, run_search(values), cfg, use_llm)

    @app.post('/ask')
    def ask_post(request: Request, payload: AskInput):
        return ask_common(request, payload)

    @app.get('/ask')
    def ask_get(request: Request, q: str = Query(min_length=1, max_length=1000), limit_: int = Query(default=10, alias='limit', ge=1, le=20),
                rerank: Literal['auto', 'always', 'never'] = 'auto', source_id: str | None = None,
                date_from: date | None = None, date_to: date | None = None, use_llm: bool = True):
        return ask_common(request, AskInput(q=q, limit=limit_, rerank=rerank, source_id=source_id, date_from=date_from, date_to=date_to, use_llm=use_llm))

    @app.post('/admin/login')
    def login(request: Request, response: Response, payload: LoginInput):
        limit(request, 'login', 5)
        expected = hashlib.sha256(cfg.admin_password.encode()).digest()
        supplied = hashlib.sha256(payload.password.encode()).digest()
        if not cfg.admin_password or not hmac.compare_digest(expected, supplied):
            raise HTTPException(401, 'Неверный пароль')
        csrf = secrets.token_urlsafe(32)
        response.set_cookie('rag_admin', serializer.dumps({'role': 'admin', 'csrf': csrf}), max_age=43200,
                            httponly=True, secure=cfg.secure_cookie, samesite='strict', path='/')
        return {'csrf': csrf}

    @app.get('/admin/session')
    def session(value=Depends(admin)):
        return {'csrf': value['csrf']}

    @app.post('/admin/logout')
    def logout(response: Response, value=Depends(admin)):
        response.delete_cookie('rag_admin', path='/')
        return {'ok': True}

    @app.get('/admin/status', dependencies=[Depends(admin)])
    def status():
        pointer = cfg.data_dir / 'indices' / 'current.json'
        index = None
        if pointer.exists():
            import json
            version = json.loads(pointer.read_text(encoding='utf-8'))['version']
            index = json.loads((pointer.parent / version / 'manifest.json').read_text(encoding='utf-8'))
        return {'documents': len(db.documents()), 'revision': db.revision(), 'index': index,
                'models': models.status(), 'llm': {'configured': bool(cfg.api_key and cfg.api_url), 'model': cfg.api_model},
                'limits': {'ask_per_minute': cfg.ask_rate, 'search_per_minute': cfg.search_rate}}

    @app.get('/admin/sources', dependencies=[Depends(admin)])
    def sources():
        return db.sources()

    @app.post('/admin/sources', dependencies=[Depends(admin)], status_code=201)
    def source_add(payload: SourceInput):
        try:
            url = validate_public_url(payload.url)
            return db.add_source(payload.name.strip(), payload.kind, url)
        except sqlite3.IntegrityError:
            raise HTTPException(409, 'Этот источник уже добавлен') from None
        except ValueError as error:
            raise HTTPException(422, str(error)) from None

    @app.patch('/admin/sources/{ident}', dependencies=[Depends(admin)])
    def source_enable(ident: str, payload: EnableInput):
        if not db.enable(ident, payload.enabled):
            raise HTTPException(404, 'Источник не найден')
        db.enqueue('index')
        return db.source(ident)

    @app.get('/admin/jobs', dependencies=[Depends(admin)])
    def jobs():
        return db.jobs()

    @app.post('/admin/jobs', dependencies=[Depends(admin)], status_code=202)
    def enqueue(payload: JobInput):
        if payload.source_id and not db.source(payload.source_id):
            raise HTTPException(404, 'Источник не найден')
        try:
            return {'job_id': db.enqueue(payload.kind, payload.source_id)}
        except ValueError as error:
            raise HTTPException(429, str(error)) from None

    @app.post('/admin/upload', dependencies=[Depends(admin)], status_code=202)
    async def upload(file: UploadFile = File()):
        from pathlib import Path
        filename = Path(file.filename or 'document.txt').name
        if Path(filename).suffix.lower() not in ('.txt', '.md', '.markdown'):
            raise HTTPException(422, 'Поддерживаются TXT и Markdown в UTF-8')
        raw = await file.read(2 * 1024 * 1024 + 1)
        if len(raw) > 2 * 1024 * 1024:
            raise HTTPException(413, 'Максимальный размер файла — 2 МБ')
        try:
            text = raw.decode('utf-8-sig').strip()
        except UnicodeDecodeError:
            raise HTTPException(422, 'Сохраните файл в UTF-8') from None
        if len(text) < 35:
            raise HTTPException(422, 'В документе недостаточно текста')
        source = next((s for s in db.sources() if s['kind'] == 'upload'), None)
        if not source:
            source = db.add_source('Загруженные документы', 'upload', 'upload://local')
        digest = hashlib.sha256(raw).hexdigest()
        url = f'/documents/{digest}'
        db.upsert(source['id'], filename, url, text)
        return {'job_id': db.enqueue('index'), 'url': url}

    @app.get('/documents/{digest}')
    def document(digest: str):
        from fastapi.responses import PlainTextResponse
        row = next((d for d in db.documents() if d['url'] == '/documents/' + digest), None)
        if not row:
            raise HTTPException(404, 'Документ не найден')
        return PlainTextResponse(row['title'] + '\n\n' + row['text'])

    @app.post('/admin/compare', dependencies=[Depends(admin)])
    def compare(request: Request, payload: AskInput):
        limit(request, 'compare', 5)
        with pipeline_slot():
            params = payload.model_dump(exclude={'use_llm', 'rerank'})
            return {mode: run_search(params | {'rerank': mode}) for mode in ('never', 'always', 'auto')}

    return app


app = create_app()
