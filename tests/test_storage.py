"""Store contracts and full-data validation equivalence; no model training."""
import json
import tempfile
import unittest
from pathlib import Path

import torch

from src.data.storage import FileStore, MemoryStore, dataset_store
from src.data.two_stage import load_stage_data
from src.models.ml.data import load

ROOT = Path(__file__).resolve().parents[1]


class StoreTests(unittest.TestCase):
    def test_read_write_and_missing_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            for store in [FileStore(), MemoryStore()]:
                path = Path(tmp) / type(store).__name__ / 'nested' / 'unicode.json'
                with self.assertRaises(FileNotFoundError):
                    store.read_bytes(path)
                payload = json.dumps({'text': '\u0111\u1ec3 ki\u1ec3m tra'}, ensure_ascii=False).encode()
                store.write_bytes(path, payload)
                self.assertEqual(store.read_bytes(path), payload)
                store.write_bytes(path, b'updated')
                self.assertEqual(store.read_bytes(path), b'updated')

    def test_snapshot_does_not_read_live_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'sample'
            path.write_bytes(b'old')
            memory = MemoryStore.snapshot(tmp)
            path.write_bytes(b'new')
            self.assertEqual(memory.read_bytes(path), b'old')
            memory.write_bytes(path, b'memory only')
            self.assertEqual(path.read_bytes(), b'new')

    def test_all_real_staged_datasets_equivalent(self):
        for stage, folder in [('kfold', 'k_fold_data'), ('final', 'full_training_data')]:
            root = ROOT / 'data' / folder
            for language in ['python', 'java', 'pharo']:
                with self.subTest(stage=stage, language=language):
                    a = load_stage_data(root, language, stage)
                    b = load_stage_data(root, language, stage, store=dataset_store('memory', root / language))
                    self.assertEqual(a.texts, b.texts)
                    self.assertEqual(a.sentence_ids, b.sentence_ids)
                    self.assertEqual(a.folds, b.folds)
                    self.assertEqual(a.data_fingerprint, b.data_fingerprint)
                    self.assertEqual(a.population_fingerprint, b.population_fingerprint)
                    for field in ['labels', 'train_mask', 'validation_mask', 'test_mask']:
                        self.assertTrue(torch.equal(getattr(a, field), getattr(b, field)))

    def test_identical_validation_failure_after_memory_corruption(self):
        root = ROOT / 'data' / 'full_training_data'
        memory = MemoryStore.snapshot(root / 'python')
        path = root / 'python' / 'label_cols.json'
        memory.write_bytes(path, b'["duplicate", "duplicate"]')
        with self.assertRaisesRegex(ValueError, 'unique label'):
            load_stage_data(root, 'python', 'final', store=memory)
        self.assertEqual(len(load_stage_data(root, 'python', 'final').label_names), 5)

    def test_ml_consumes_same_injected_store(self):
        root = ROOT / 'data' / 'full_training_data'
        a = load('python', 'final', root)
        b = load('python', 'final', root, store=MemoryStore.snapshot(root / 'python'))
        self.assertEqual(a['texts'].tolist(), b['texts'].tolist())
        self.assertTrue((a['labels'] == b['labels']).all())

    def test_reject_invalid_backend(self):
        with self.assertRaisesRegex(ValueError, 'storage backend'):
            dataset_store('invalid', ROOT)


if __name__ == '__main__':
    unittest.main()
