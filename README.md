# Bounded geological residual fusion

Research code for **Bounded geological residual fusion for semi-supervised hyperspectral copper prospectivity mapping in eastern Junggar**, by Zilong Cao, Qing Wang, Wei Wei, Qinkang Ma and Xin Liu. Correspondence: Qing Wang, kidqing@yangtzeu.edu.cn. Manuscript under preparation; no publication DOI has been assigned.

The method adds map-unit embeddings and fault-distance encodings to a GAN-U-Net decoder through zero-initialised projection, channel gating and a relative RMS bound. Mean Teacher consistency uses unlabelled hyperspectral patches. Five spectral backbone implementations, geological-input controls and component ablations are included.

## What can be run

| Task | Public inputs sufficient? | Hardware |
| --- | --- | --- |
| Synthetic numerical checks | Yes; tensors are generated in memory | CPU |
| Training and DEV evaluation | No; an authorised prepared cache is required | NVIDIA CUDA with bfloat16 support |
| Prediction from a saved teacher | No; a matching checkpoint and prepared cache are required | CPU or CUDA |
| Reproduction of manuscript scores | No; original restricted data, partitions and numerical environment are required | Recorded research environment |

No research data or pretrained weights are downloaded by this package. The public example checks implementation behaviour and does not reproduce the manuscript results.

## Installation

The package declares Python >=3.10, PyTorch >=2.1,<3, NumPy >=1.24,<3 and scikit-learn >=1.3,<2. The Python version must also be supported by the particular PyTorch wheel you install. Use a separate environment; the original research versions are recorded in [Numerical protocol](docs/PROTOCOL.md#environment-and-source-fidelity).

Download and extract this repository, or clone it:

```sh
git clone https://github.com/longlonghu283-cpu/bounded-geological-residual-fusion.git
cd bounded-geological-residual-fusion
```

For **Windows PowerShell**, the following installs a CPU environment without requiring activation scripts:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install "torch>=2.1,<3" --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe examples/synthetic_smoke.py
```

For **Linux**, a CPU environment can be prepared with:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "torch>=2.1,<3" --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e .
python examples/synthetic_smoke.py
```

For CUDA training or a different platform, replace the CPU installation command with the compatible command from the [official PyTorch installation selector](https://pytorch.org/get-started/locally/), then install this package with `python -m pip install -e .`. The package does not require torchvision or torchaudio. Installation requires network access; the synthetic example needs none after installation.

All later commands use `python` to mean the interpreter in this environment. In PowerShell, use `.\.venv\Scripts\python.exe` in its place if the environment has not been activated.

## Check the installation

```sh
python examples/synthetic_smoke.py
python -m georesidual.train --help
```

The first command prints a JSON report containing `"status": "PASS"` if all assertions succeed. It checks five spectral backbones, zero-initialisation identity, the HSI bypass, geological masks, the RMS bound, two component switches, finite gradients, and agreement of AP and threshold calculations. It performs no optimiser updates and does not create research files. A CPU with 8 GB RAM is recommended. See [Recorded verification](docs/VALIDATION.md) for the tested scope; this check does not validate the complete CUDA training cycle or raw-GIS preparation.

## Prepare inputs

Read [Input files and required fields](docs/INPUTS.md) before training. The public package consumes registered raster caches; it does not include a raw-imagery or shapefile preprocessing command. The cache directory must contain the required arrays and per-fold metadata. Training, normalisation and DEV-support definitions are specified in that document and must be prepared consistently.

E3 uses the spectral cache. Every E4 variant, including the HSI-only repeat, currently loads the geological cache as well. The HSI-only repeat bypasses geology during forward computation, rather than bypassing cache loading.

## Train one configuration

These commands start real training and require authorised prepared data plus a CUDA device supporting bfloat16. Paths below are examples, not bundled datasets. The output directory must not already exist.

```sh
python -m georesidual.train --stage e3 --model gan_unet --seed 42 --fold 0 --data data/cache --out outputs/e3_gan_f0_s42
python -m georesidual.train --stage e4 --variant joint --seed 42 --fold 0 --data data/cache --out outputs/e4_joint_f0_s42
```

| Argument | Allowed values or meaning |
| --- | --- |
| `--stage` | `e3` or `e4`; required |
| `--model` | E3 only: `unet`, `gan_unet`, `gcn_unet`, `convlstm_unet`, `segformer_b0`; default `gan_unet` |
| `--variant` | E4 only; configurations listed below; default `joint` |
| `--seed` | `42`, `7` or `2026`; required |
| `--fold` | `0`, `1`, `2` or `3`; required |
| `--data` | Prepared cache directory; required |
| `--out` | A new run directory; required |

E4 always uses the GAN-U-Net backbone; `--model` does not select an E4 backbone. Conversely, `--variant` has no effect on E3. Do not use these ignored arguments to label an experiment.

| E3 identifier | Model |
| --- | --- |
| `unet` | MT-U-Net |
| `gan_unet` | GAN-U-Net, with a mask discriminator during training |
| `gcn_unet` | MT-GCN-U-Net |
| `convlstm_unet` | MT-ConvLSTM-U-Net |
| `segformer_b0` | MT-SegFormer-B0 |

| E4 identifier | Configuration |
| --- | --- |
| `joint` | Complete geological fusion |
| `hsi` | Paired HSI-only repeat; geological forward branch bypassed |
| `strata` | Map units retained; normalised fault distance set to zero |
| `fault` | Fault distance retained; valid category indices set to 1 |
| `noise` | Active capacity-matched branch with pseudorandom attributes and original coverage mask |
| `no_gate` | Unit channel gate |
| `nonzero_init` | Conventional projection initialisation |
| `no_cap` | Relative RMS bound removed |

Each invocation performs 300 updates and evaluates every 30 updates. A full configuration comprises the twelve combinations of three seeds and four folds, each in a separate output directory. The wrapper has no automatic resume function. See [Usage details](docs/USAGE.md) for outputs, checkpoint loading, inference and troubleshooting.

## Read the results

| Run product | Meaning |
| --- | --- |
| `updates.jsonl` | Per-update losses, gradients, learning rate and diagnostics |
| `result.json` | Progress at the last evaluation and metrics of the best selected teacher so far |
| `selected_teacher.pt` | Selected teacher's parameter state; not a resumable training checkpoint |
| `selected_DEV.npz` | Sorted global `keys`, binary `labels` and `probabilities` at scored DEV positions |
| `controls/FOLDk.npz` | Pseudorandom input fields, produced only for `noise`; retain for matching inference |

Check `completed_updates == 300` in `result.json` before treating a run as complete. The selected checkpoint maximises DEV AP, with validation loss and then earlier update breaking ties. It need not be the last update. AP uses average precision; threshold-dependent metrics use the DEV threshold maximising F1. Aggregation and paired differences are defined in [Numerical protocol](docs/PROTOCOL.md).

## Repository files

```text
bounded-geological-residual-fusion/
  README.md
  LICENSE
  CITATION.cff
  pyproject.toml
  requirements.txt
  .gitignore
  georesidual/
    __init__.py
    models.py
    backbone.py
    fusion.py
    e3_engine.py
    e4_engine.py
    controls.py
    train.py
    inference.py
  examples/
    synthetic_smoke.py
  docs/
    INPUTS.md
    PROTOCOL.md
    USAGE.md
    VALIDATION.md
```

`models.py` contains the five E3 choices; `backbone.py` and `fusion.py` define E4. The engines retain scientific sampling, augmentation, losses, optimisation, EMA and DEV evaluation functions. `controls.py` implements ablations, `train.py` exposes the opt-in runner, and `inference.py` averages overlapping prediction windows. `requirements.txt` lists dependencies; `pyproject.toml` also installs the package itself.

## Data and reproducibility

Research imagery, geological vectors, occurrence coordinates, cached rasters, trained checkpoints, prediction vectors and original run logs are excluded because of confidentiality and redistribution restrictions. Code is released under MIT; this licence does not grant access to or redistribution rights for research data. The public example verifies implementation behaviour, not the reported performance. Reproduction of manuscript numbers requires the authorised original data, frozen partitions and numerical environment. Application to new data is a new experiment.

Reported metrics are within-scene development results. Radius, configuration, checkpoint and threshold choices use development evaluation. They do not establish performance on independent unknown mineral occurrences. The paired HSI repeat and active capacity-matched noise-attribute control answer different questions; see [Control definitions](docs/PROTOCOL.md#meaning-of-controls).

## Citation and support

Software citation metadata are provided in [CITATION.cff](CITATION.cff). Cite the version or commit used in an application. The manuscript is under preparation; no accepted-publication status or DOI is claimed here. Use repository issues for code questions without including private data, coordinates, paths, checkpoints or logs. Scientific correspondence: Qing Wang, kidqing@yangtzeu.edu.cn.

Documentation updated on 13 November 2026. Scientific implementation and experiment settings are unchanged by this documentation revision.
