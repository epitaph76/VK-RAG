import os
import threading
import time
import uuid

from filelock import FileLock

from .config import Settings
from .db import Database, now
from .index import build_index
from .ingest import collect
from .models import Models


def run_job(db, models, cfg, job):
    ident = job['id']
    results, errors = [], []
    if job['kind'] == 'ingest':
        sources = [s for s in db.sources(public=True) if s['kind'] != 'upload' and (not job['source_id'] or s['id'] == job['source_id'])]
        for source in sources:
            try:
                report = collect(db, source, cfg.source_limit,
                                 lambda done, total, report: db.job_update(ident, progress=done, total=total, result=report))
                results.append({'source': source['name'], **report})
                source_error = f"Не загружено страниц: {report['skipped']}" if report['skipped'] else None
                if source_error:
                    errors.append({'source': source['name'], 'error': source_error})
            except Exception as error:
                source_error = str(error)[:200]
                errors.append({'source': source['name'], 'error': source_error})
            with db.connect() as connection:
                connection.execute('UPDATE sources SET last_run=?,last_error=? WHERE id=?', (now(), source_error, source['id']))
    manifest = build_index(db, models, cfg, lambda done, total: db.job_update(ident, progress=done, total=total))
    state, message = 'done', None
    if errors:
        if any(report['fetched'] for report in results):
            state, message = 'partial', 'Обновление выполнено частично. Подробности по источникам ниже.'
        else:
            state, message = 'failed', 'Не удалось загрузить статьи. Ранее сохранённые материалы остаются доступны.'
    db.job_update(ident, state=state, finished_at=now(), result={'sources': results, 'errors': errors, 'index': manifest}, error=message)


def main():
    cfg = Settings()
    db = Database(cfg.database)
    db.seed()
    models = Models(cfg)
    with FileLock(str(cfg.data_dir / 'worker.lock'), timeout=0):
        # Only this process can own the queue; resume interrupted jobs after its predecessor exited.
        with db.connect() as connection:
            connection.execute("UPDATE jobs SET state='queued',worker_id=NULL,error='Продолжение после перезапуска' WHERE state='running'")
        worker_id = uuid.uuid4().hex
        while True:
            job = db.claim(worker_id)
            if not job:
                time.sleep(2)
                continue
            stop = threading.Event()
            def heartbeat():
                while not stop.wait(10):
                    db.job_update(job['id'], heartbeat=now())
            thread = threading.Thread(target=heartbeat, daemon=True)
            thread.start()
            try:
                run_job(db, models, cfg, job)
            except Exception:
                db.job_update(job['id'], state='failed', finished_at=now(), error='Сбор или индексирование завершились ошибкой; предыдущий индекс сохранён')
            finally:
                stop.set()
                thread.join()


if __name__ == '__main__':
    main()
