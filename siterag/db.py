import hashlib
import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone


def now():
    return datetime.now(timezone.utc).isoformat()


SCHEMA = """
CREATE TABLE IF NOT EXISTS sources (
 id TEXT PRIMARY KEY, name TEXT NOT NULL, kind TEXT NOT NULL,
 url TEXT NOT NULL UNIQUE, enabled INTEGER NOT NULL DEFAULT 1,
 created_at TEXT NOT NULL, last_run TEXT, last_error TEXT);
CREATE TABLE IF NOT EXISTS contents (
 hash TEXT PRIMARY KEY, text TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS documents (
 id TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id),
 title TEXT NOT NULL, url TEXT NOT NULL UNIQUE, content_hash TEXT NOT NULL REFERENCES contents(hash),
 published_at TEXT, fetched_at TEXT NOT NULL);
CREATE INDEX IF NOT EXISTS documents_source ON documents(source_id);
CREATE TABLE IF NOT EXISTS jobs (
 id TEXT PRIMARY KEY, kind TEXT NOT NULL, source_id TEXT REFERENCES sources(id),
 state TEXT NOT NULL, created_at TEXT NOT NULL, started_at TEXT, finished_at TEXT,
 progress INTEGER NOT NULL DEFAULT 0, total INTEGER NOT NULL DEFAULT 0,
 result TEXT NOT NULL DEFAULT '{}', error TEXT, worker_id TEXT, heartbeat TEXT);
CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
INSERT OR IGNORE INTO metadata VALUES ('revision', '0');
"""


class Database:
    def __init__(self, path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript(SCHEMA)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA journal_mode=WAL")
        try:
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def seed(self):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            # Migrate the old default once; keep its documents and user-created sources.
            if not db.execute("SELECT 1 FROM metadata WHERE key='source_defaults_v2'").fetchone():
                changed = db.execute("""UPDATE sources SET enabled=0
                    WHERE name=? AND url=? AND enabled=1""",
                    ("N+1 · наука", "https://nplus1.ru/rss")).rowcount
                if changed:
                    self.bump(db)
                db.execute("INSERT INTO metadata VALUES ('source_defaults_v2', '1')")
            for name, url in [
                ("Хабр · статьи", "https://habr.com/ru/rss/articles/?fl=ru"),
                ("Элементы · наука", "https://elementy.ru/rss/news"),
            ]:
                db.execute("INSERT OR IGNORE INTO sources(id,name,kind,url,created_at) VALUES(?,?,?,?,?)",
                           (uuid.uuid4().hex, name, "rss", url, now()))

    def sources(self, public=False):
        with self.connect() as db:
            rows = db.execute("SELECT * FROM sources" + (" WHERE enabled=1" if public else "") + " ORDER BY created_at").fetchall()
            return [dict(r) for r in rows]

    def source(self, source_id):
        with self.connect() as db:
            row = db.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()
            return dict(row) if row else None

    def add_source(self, name, kind, url):
        ident = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("INSERT INTO sources(id,name,kind,url,created_at) VALUES(?,?,?,?,?)", (ident, name, kind, url, now()))
        return self.source(ident)

    def enable(self, ident, enabled):
        with self.connect() as db:
            changed = db.execute("UPDATE sources SET enabled=? WHERE id=?", (int(enabled), ident)).rowcount
            if changed:
                self.bump(db)
        return changed

    @staticmethod
    def bump(db):
        db.execute("UPDATE metadata SET value=CAST(value AS INTEGER)+1 WHERE key='revision'")

    def revision(self):
        with self.connect() as db:
            return int(db.execute("SELECT value FROM metadata WHERE key='revision'").fetchone()[0])

    def upsert(self, source_id, title, url, text, published_at=None):
        digest = hashlib.sha256(text.encode()).hexdigest()
        ident = hashlib.sha256(url.encode()).hexdigest()[:24]
        with self.connect() as db:
            old = db.execute("SELECT content_hash,title,published_at,source_id FROM documents WHERE url=?", (url,)).fetchone()
            changed = not old or tuple(old) != (digest, title, published_at, source_id)
            db.execute("INSERT OR IGNORE INTO contents(hash,text) VALUES(?,?)", (digest, text))
            db.execute("""INSERT INTO documents VALUES(?,?,?,?,?,?,?) ON CONFLICT(url) DO UPDATE SET
                       title=excluded.title, content_hash=excluded.content_hash, published_at=excluded.published_at,
                       fetched_at=excluded.fetched_at, source_id=excluded.source_id""",
                       (ident, source_id, title, url, digest, published_at, now()))
            if changed:
                self.bump(db)
        return changed

    def documents(self):
        with self.connect() as db:
            return [dict(r) for r in db.execute("""SELECT d.*, c.text, s.name source_name FROM documents d
                JOIN contents c ON c.hash=d.content_hash JOIN sources s ON s.id=d.source_id
                WHERE s.enabled=1 ORDER BY d.id""")]

    def snapshot(self):
        with self.connect() as db:
            db.execute("BEGIN")
            revision = int(db.execute("SELECT value FROM metadata WHERE key='revision'").fetchone()[0])
            docs = [dict(r) for r in db.execute("""SELECT d.*, c.text, s.name source_name FROM documents d
                JOIN contents c ON c.hash=d.content_hash JOIN sources s ON s.id=d.source_id
                WHERE s.enabled=1 ORDER BY d.id""")]
        return revision, docs

    def enqueue(self, kind="ingest", source_id=None):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT COUNT(*) FROM jobs WHERE state IN ('queued','running')").fetchone()[0] >= 8:
                raise ValueError("Очередь сбора заполнена")
            existing = db.execute("SELECT id FROM jobs WHERE kind=? AND source_id IS ? AND state IN ('queued','running')", (kind, source_id)).fetchone()
            if existing:
                return existing[0]
            ident = uuid.uuid4().hex
            db.execute("INSERT INTO jobs(id,kind,source_id,state,created_at) VALUES(?,?,?,'queued',?)", (ident, kind, source_id, now()))
        return ident

    def claim(self, worker_id):
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT * FROM jobs WHERE state='queued' ORDER BY created_at LIMIT 1").fetchone()
            if not row:
                return None
            db.execute("UPDATE jobs SET state='running',started_at=?,worker_id=?,heartbeat=? WHERE id=?", (now(), worker_id, now(), row['id']))
            return dict(row)

    def job_update(self, ident, **values):
        allowed = {'state', 'progress', 'total', 'result', 'error', 'finished_at', 'heartbeat'}
        if not values or not set(values) <= allowed:
            raise ValueError("Invalid job update")
        values = {k: json.dumps(v, ensure_ascii=False) if k == 'result' else v for k, v in values.items()}
        with self.connect() as db:
            db.execute("UPDATE jobs SET " + ",".join(k + "=?" for k in values) + " WHERE id=?", (*values.values(), ident))

    def jobs(self):
        with self.connect() as db:
            rows = [dict(r) for r in db.execute("SELECT * FROM jobs ORDER BY created_at DESC LIMIT 40")]
        for row in rows:
            row['result'] = json.loads(row['result'])
        return rows
