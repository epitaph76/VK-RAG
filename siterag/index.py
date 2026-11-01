import hashlib
import json
import os
import re
import uuid
from pathlib import Path

import numpy as np
from filelock import FileLock


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')
    os.replace(temp, path)


def hash_file(path):
    with open(path, 'rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def tokens(text):
    return re.findall(r'\w+', text.casefold().replace('ё', 'е'))


def chunk_documents(documents, tokenizer=None, size=384, overlap=64):
    if not 0 <= overlap < size:
        raise ValueError('Invalid chunk overlap')
    chunks = []
    for doc in documents:
        encoded = tokenizer.encode(doc['text'], add_special_tokens=False, truncation=False, verbose=False) if tokenizer else doc['text'].split()
        for start in range(0, len(encoded), size - overlap):
            part = encoded[start:start + size]
            text = tokenizer.decode(part, skip_special_tokens=True) if tokenizer else ' '.join(part)
            chunks.append({k: v for k, v in doc.items() if k != 'text'} | {
                'id': f"{doc['id']}:{start}", 'document_id': doc['id'], 'text': text,
            })
            if start + size >= len(encoded):
                break
    return chunks


def build_index(database, models, settings, progress=None):
    import bm25s
    root = settings.data_dir / 'indices'
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / 'build.lock')):
        revision, documents = database.snapshot()
        warnings = []
        encoder = None
        try:
            encoder = models.get('encoder')
        except Exception:
            warnings.append('Dense недоступен: создан индекс BM25 с чанкингом по словам')
        chunks = chunk_documents(documents, encoder.tokenizer if encoder else None, settings.chunk_tokens, settings.overlap)
        version = f'{revision}-{uuid.uuid4().hex[:12]}'
        path = root / version
        path.mkdir()
        atomic_json(path / 'chunks.json', chunks)
        if chunks:
            index = bm25s.BM25()
            index.index(bm25s.tokenize([c['title'] + '\n' + c['text'] for c in chunks], stopwords=None, show_progress=False), show_progress=False)
            index.save(str(path / 'bm25'))
        dense = False
        if encoder and chunks:
            try:
                import faiss
                faiss.omp_set_num_threads(settings.threads)
                batches = []
                for start in range(0, len(chunks), 24):
                    batch = ['passage: ' + c['text'] for c in chunks[start:start + 24]]
                    batches.append(encoder.encode(batch, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False))
                    if progress:
                        progress(min(start + 24, len(chunks)), len(chunks))
                vectors = np.concatenate(batches).astype('float32')
                np.save(path / 'vectors.npy', vectors)
                faiss_index = faiss.IndexFlatIP(vectors.shape[1])
                faiss_index.add(vectors)
                faiss.write_index(faiss_index, str(path / 'dense.faiss'))
                dense = True
            except Exception:
                warnings.append('Построение Dense не удалось: опубликован рабочий BM25')
        corpus_hash = hashlib.sha256(json.dumps([(d['id'], d['content_hash']) for d in documents], sort_keys=True).encode()).hexdigest()
        manifest = {'version': version, 'revision': revision, 'documents': len(documents), 'chunks': len(chunks),
                    'corpus_hash': corpus_hash, 'dense': dense, 'warnings': warnings,
                    'chunking': {'size': settings.chunk_tokens, 'overlap': settings.overlap, 'unit': 'tokens' if encoder else 'words'},
                    'models': dict(models.revisions), 'files': {p.relative_to(path).as_posix(): hash_file(p) for p in path.rglob('*') if p.is_file()}}
        atomic_json(path / 'manifest.json', manifest)
        # Only this final pointer switches readers. Incomplete builds stay invisible.
        atomic_json(root / 'current.json', {'version': version})
        return manifest


class Snapshot:
    def __init__(self, root, version):
        import bm25s
        if not re.fullmatch(r'\d+-[a-f0-9]{12}', version):
            raise ValueError('Invalid snapshot name')
        path = root / version
        self.manifest = json.loads((path / 'manifest.json').read_text(encoding='utf-8'))
        for filename, digest in self.manifest['files'].items():
            if not (path / filename).resolve().is_relative_to(path.resolve()) or hash_file(path / filename) != digest:
                raise ValueError('Повреждён индекс; выполните перестроение')
        self.chunks = json.loads((path / 'chunks.json').read_text(encoding='utf-8'))
        self.bm25 = bm25s.BM25.load(str(path / 'bm25'), load_corpus=False) if self.chunks else None
        self.vectors = np.load(path / 'vectors.npy', mmap_mode='r') if self.manifest['dense'] else None
        self.faiss = None
        if self.manifest['dense']:
            import faiss
            self.faiss = faiss.read_index(str(path / 'dense.faiss'))
