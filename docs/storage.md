# Staged dataset persistence

`src/data/storage.py` provides a `ByteStore` protocol with `read_bytes(path)` and `write_bytes(path, value)`. `FileStore` reads and persists files. `MemoryStore` owns an isolated byte snapshot with the same path keys and missing-key exception contract.

`load_stage_data(..., store=...)` performs identical parsing, split validation, mask construction and fingerprinting for both stores. The ML adapter accepts the same injected store. Both staged experiment families select the backend at their composition boundary; model/feature/evaluation logic is unchanged.

```python
from pathlib import Path
from src.data.storage import FileStore, MemoryStore
from src.data.two_stage import load_stage_data

root = Path("data/full_training_data")
files = load_stage_data(root, "python", "final", store=FileStore())
snapshot = MemoryStore.snapshot(root / "python")
memory = load_stage_data(root, "python", "final", store=snapshot)
assert files.data_fingerprint == memory.data_fingerprint
```

Both Python training CLIs accept `--storage file` (default) or `--storage memory`:

```bash
python -m experiments.ml.train_baseline --stage kfold --language python --storage memory --output-dir results/new_ml_memory
python -m experiments.dl.train_exp1 --stage kfold --language python --storage memory --device cpu --output-dir results/new_dl_memory
```

These are training commands, not smoke tests. Use fresh output directories. Memory mode initially reads the prepared language directory into a snapshot, then the staged loader and data-manifest copy read from that snapshot. Changing live files after snapshot construction does not change the snapshot; memory writes do not change files.

This interface currently covers **staged dataset persistence**. Run metrics, plots, tokenizer/checkpoint files and serialized models continue to export through explicit filesystem directories. The historical research runs predate the adapters and have not been retrained.

Validation without model training:

```bash
python -m unittest discover -s tests -p test_storage.py -v
```

Six tests verify byte round trips, missing paths, snapshot isolation, validation failures, ML consumption, and data/mask/fold/fingerprint equivalence across all six real language/stage datasets. The complete offline suite also exercises the existing staged runners with tiny mocked models.
