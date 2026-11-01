import os
import threading
from pathlib import Path

os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


class Models:
    def __init__(self, settings):
        self.settings = settings
        self.instances = {}
        self.failures = {}
        self.revisions = {}
        self.lock = threading.RLock()

    def snapshot(self, repo, revision):
        from huggingface_hub import snapshot_download
        required = ['config.json', 'model.safetensors']
        required += ['encoder/config.json', 'tokenizer/tokenizer.json', 'rl_agent_config.json'] if repo == self.settings.laya_model else ['tokenizer.json']
        if repo == self.settings.embedding_model:
            required += ['modules.json', '1_Pooling/config.json']
        try:
            path = snapshot_download(repo, revision=revision, local_files_only=True)
            if not all((Path(path) / name).is_file() for name in required):
                raise FileNotFoundError('Incomplete model cache')
        except (FileNotFoundError, OSError):
            path = snapshot_download(repo, revision=revision,
                                     allow_patterns=['*.json', '*.safetensors', '*.txt', '*.model'])
        self.revisions[repo] = path.replace('\\', '/').rstrip('/').split('/')[-1]
        return path

    def get(self, kind):
        with self.lock:
            if not self.settings.models_enabled:
                raise RuntimeError('Локальные модели отключены')
            if kind in self.failures:
                raise RuntimeError(self.failures[kind])
            if kind in self.instances:
                return self.instances[kind]
            try:
                import torch
                torch.set_num_threads(self.settings.threads)
                cfg = self.settings
                if kind == 'encoder':
                    from sentence_transformers import SentenceTransformer
                    instance = SentenceTransformer(self.snapshot(cfg.embedding_model, cfg.embedding_revision), device='cpu', trust_remote_code=False)
                    instance.max_seq_length = 512
                elif kind == 'gate':
                    from laya import Router
                    instance = Router(models={'multilingual': self.snapshot(cfg.laya_model, cfg.laya_revision)}, device='cpu', max_loaded=1)
                elif kind == 'reranker':
                    from sentence_transformers import CrossEncoder
                    instance = CrossEncoder(self.snapshot(cfg.reranker_model, cfg.reranker_revision), device='cpu', max_length=512, trust_remote_code=False)
                else:
                    raise ValueError('Unknown model')
                self.instances[kind] = instance
                return instance
            except Exception:
                self.failures[kind] = f'Модель {kind} недоступна; проверьте кеш и соединение, затем перезапустите процесс'
                raise RuntimeError(self.failures[kind]) from None

    def status(self):
        return {'loaded': list(self.instances), 'errors': dict(self.failures), 'revisions': dict(self.revisions)}
