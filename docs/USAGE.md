# Running and interpreting the public implementation

Start with the installation commands in [README](../README.md). This page describes the existing command-line interface and Python inference interface. It does not provide confidential data or claim that the public example reproduces the paper's scores.

## 1. Choose the task

- To check the installation without research data, run `python examples/synthetic_smoke.py`.
- To train, first prepare an authorised cache matching [INPUTS.md](INPUTS.md), then use `python -m georesidual.train`.
- To predict from a teacher checkpoint, use the Python examples below. The repository does not expose a separate inference command-line parser.

Use a new output directory for every stage, configuration, fold and seed. Each process should run one configuration; the engines use module-level data paths and control settings. The package does not launch all experiments automatically.

## 2. Training and completion

The CLI fixes 300 updates, evaluates after updates 30, 60, ..., 300, and uses the settings in [PROTOCOL.md](PROTOCOL.md). There are no CLI arguments for epochs, learning rate, update budget or resuming. Changing those constants creates a different protocol.

For example, the following is one of twelve runs for the joint E4 configuration:

```sh
python -m georesidual.train --stage e4 --variant joint --seed 7 --fold 3 --data data/cache --out outputs/e4_joint_f3_s7
```

Repeat explicitly for seeds 42, 7 and 2026 and folds 0-3 when a full configuration is required. Prepare all corresponding fold files first. Run commands only when training is intended; `--help` displays options without starting training.

`result.json` is rewritten after each evaluation. Its `completed_updates` records the last completed evaluation point, while `selected.update` identifies the selected teacher. A run is complete when `completed_updates` is 300 and it exited successfully. If an interruption occurs between evaluations, `updates.jsonl` may extend beyond the progress stored in `result.json`. The wrapper will not continue an existing directory. Retain interrupted outputs for diagnosis; use a new directory for a deliberate new run.

The checkpoint contains only the teacher's `state_dict`. It does not contain the student, optimiser, scheduler or RNG states needed for exact training recovery. The final-step teacher is saved only if it is also the selected teacher.

`result.json` includes both `model` and `variant` because the parser accepts both flags. Only `model` is used for E3; only `variant` is used for E4. The actual stage must be considered when interpreting those fields.

## 3. Inspect selected DEV predictions

This example reads a completed authorised run; it does not generate predictions or train a model:

```python
from pathlib import Path
import json
import numpy as np

run = Path("outputs/e4_joint_f0_s42")
record = json.loads((run / "result.json").read_text(encoding="utf-8"))
assert record["completed_updates"] == 300
print(record["selected"])

with np.load(run / "selected_DEV.npz", allow_pickle=False) as saved:
    keys = saved["keys"]
    labels = saved["labels"]
    probabilities = saved["probabilities"]
    assert keys.ndim == labels.ndim == probabilities.ndim == 1
    assert len(keys) == len(labels) == len(probabilities)
    print("Scored pixels:", len(keys))
```

`keys` are global row-times-width-plus-column indices. They are sorted and deduplicated; overlapping patch predictions have been averaged at these positions. The associated `labels` are DEV labels, not all scene pixels or the complete R225 training-label raster.

For a configuration mean, average its twelve run-level metric values equally. For seed variability, first average four folds within each seed, then compute the sample SD of those three means. For a paired delta, subtract the matching control run within the same fold and seed before either aggregation. Missing runs must not silently be treated as zeros or replaced by manuscript values.

## 4. Load an E3 teacher and predict the scene

Use the same model identifier and fold as the saved checkpoint. This example uses GAN-U-Net and its spectral cache:

```python
from pathlib import Path
import numpy as np
import torch
from georesidual import e3_engine
from georesidual.models import create_model
from georesidual.inference import predict_scene

cache = Path("data/cache").resolve()
run = Path("outputs/e3_gan_f0_s42")
e3_engine.DATA = cache
data = e3_engine.RealData(fold=0)

model = create_model("gan_unet", in_channels=158)
state = torch.load(run / "selected_teacher.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state)
scores = predict_scene(model, data, device="cpu", stride=32, geology=False)
np.save(run / "scene_scores.npy", scores)
```

`geology=False` is required for E3. The spectral data adapter returns three arrays per patch, whereas the E4 adapter supplies three additional geological arrays.

## 5. Load an E4 teacher with its original variant

The checkpoint's variant and fold must match the data adapter and model flags:

```python
from pathlib import Path
import numpy as np
import torch
from georesidual import e4_engine, controls
from georesidual.backbone import UNet
from georesidual.fusion import C2GANUNet
from georesidual.inference import predict_scene

cache = Path("data/cache").resolve()
run = Path("outputs/e4_joint_f0_s42").resolve()
fold, variant = 0, "joint"
e4_engine.DATA = cache
controls.ROOT = run
data = controls.VariantData(fold, variant)

model = C2GANUNet(UNet(158), data.lock["n_embeddings"])
state = torch.load(run / "selected_teacher.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.geo.variant = variant
model.geo_enabled = variant != "hsi"
scores = predict_scene(model, data, device="cpu", stride=32, geology=True)
np.save(run / "scene_scores.npy", scores)
```

For other E4 conditions, change `variant`, `run` and `fold` to the matching values. The HSI repeat still uses this E4 model and cache interface, although `geo_enabled=False` bypasses its geological branch. Noise-control inference also requires the exact `controls/FOLDk.npz` generated in that run directory.

After loading a checkpoint, set flags as shown; do not call `apply_variant(..., "nonzero_init")`. That training-initialisation function would replace the loaded projection weights with fresh initial weights.

## 6. Understand the scene output

`predict_scene` returns a NumPy `float32` array with shape `(H, W)`. It averages sigmoid scores over overlapping 64 x 64 windows; stride defaults to 32 and must lie between 1 and 64. Both scene dimensions must be at least 64. Pixels outside the image-valid mask are `NaN`.

The function does not attach a CRS, transform or geographic metadata. Saving `scene_scores.npy` therefore preserves array positions only. A separately authorised georeferencing source is needed for map export. The same fold-specific spectral normalisation and geological vocabulary must be retained when using a checkpoint; fitting them anew changes its input representation.

These examples use ordinary FP32 inference on every valid scene pixel, including training regions. The training engines instead evaluate BF16 predictions over fixed DEV windows and scoring support. A full-scene map is not an independent validation result, and its numeric values need not be bitwise identical to saved DEV outputs. Use `selected_DEV.npz` and `result.json` for the selected-run DEV result. This helper does not reproduce the paper's map styling or compute DEV metrics automatically.

## 7. Troubleshooting

| Symptom | Check or action |
| --- | --- |
| `No module named georesidual` | Run `python -m pip install -e .` from the repository root with the same interpreter used to run commands. |
| PyTorch cannot be installed | Select a supported Python, OS and compute build using the official PyTorch installer; do not assume every Python version has every CUDA wheel. |
| `Frozen training requires CUDA with bfloat16 support` | A CPU-only build, unavailable GPU or unsupported BF16 device cannot run this frozen training wrapper. The synthetic check can still run on CPU. |
| Output directory already exists | Select a new directory. Existing outputs are not overwritten or resumed. |
| Missing array, fold JSON or NPZ key | Compare the cache against the exact filenames and keys in [INPUTS.md](INPUTS.md). |
| `EVALUATION_SUPPORT_CHANGED` or key-equality assertion | Recheck sorted deduplicated DEV keys, the support mask and `val_P` windows. Do not disable the assertion. |
| Embedding index/size mismatch on loading or prediction | Check the checkpoint's fold, category vocabulary and `n_embeddings`; do not rebuild a vocabulary from DEV data. |
| Non-finite values | Check HSI fill values, positive standard deviations, distance values and common-grid alignment. Multiplication by zero does not remove NaNs. |
| Noise-control fields missing | Retain the original run's `controls/FOLDk.npz` and point `controls.ROOT` at that run. |
| CUDA out of memory | This wrapper fixes its batch and patch sizes. Use sufficient hardware; changing these sizes would define a different experiment. |

## 8. Validation and confidentiality

The recorded CPU checks are listed in [VALIDATION.md](VALIDATION.md). The usage snippets were checked against the released interfaces; a full-data training or inference run has not been rerun for this documentation update. Original research inputs and outputs remain restricted. Keep caches, checkpoints, coordinate arrays and run products out of the public repository.
