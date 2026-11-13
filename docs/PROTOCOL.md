# Numerical protocol and controls

- 158 input bands; 64 x 64 pixels per window; 4 labelled and 8 unlabelled windows per update.
- Seeds 42, 7 and 2026; four development folds; 300 updates; evaluation every 30 updates.
- AdamW: learning rate 1e-4, weight decay 1e-4, betas (0.9,0.999), epsilon 1e-8, foreach/fused disabled. Cosine decay over 300 updates.
- Weighted BCE plus Dice on labelled pixels; EMA teacher decay 0.99; confident unlabelled teacher probabilities >=0.7 or <=0.3; consistency ramp 0.2*exp(-5*(1-min((update-1)/180,1))^2).
- Weak Gaussian spectral noise SD 0.01. Strong multiplicative factor 0.9–1.1, noise SD 0.05 and band dropout 0.08. Rotations/flips are shared across registered modalities.
- GAN mask adversarial coefficient 0.01. E4 map-unit embedding 8, geological encoder 16, injected decoder feature 64 channels. Relative RMS bound 0.01; epsilon 1e-8; bound arithmetic FP32.
- Checkpoint ranking: greatest DEV AP, then least DEV supervised loss, then earliest update. Classification threshold maximises DEV F1; ties select the higher threshold. AP is average precision, not trapezoidal integration of the PR curve.
- Mean metrics weight each fold-seed run equally. Seed SD is sample SD (ddof=1) of three four-fold means. Paired differences subtract matched runs before aggregation.

## Meaning of controls

`hsi` bypasses the geological branch. Stored branch parameters remain allocated, but effective model capacity changes. This is the matched-initialisation/training-sequence HSI repeat, not the original E3 benchmark and not an active capacity-matched control.

`noise` retains the active geological branch and original coverage mask, replacing valid map-unit codes with pseudorandom categories and normalised fault distance with pseudorandom standard-normal values. Fixed control-input seed stream is derived from 20261025 and fold. It matches architecture and coverage while removing real geological attributes. It is not a control for the information contained in the coverage mask itself.

`strata` zeros normalised fault distance; `fault` uses index 1 for all valid categories while retaining distance. `no_gate` uses a unit gate. `nonzero_init` restores the constructor's conventional projection weights. `no_cap` forwards the raw residual. Joint input uses all components. The original paired training implementation isolates geological initialisation from the backbone/discriminator and sampling RNG streams.

## Environment and source fidelity

Recorded research environment: Python 3.10.20, PyTorch 2.11.0+cu128, NumPy 2.2.6, scikit-learn 1.5.0, CUDA 12.8, cuDNN 91900, NVIDIA RTX 5060 Laptop GPU. Four CPU threads; cuDNN benchmark off; matmul TF32 off and cuDNN TF32 on. E3 deterministic algorithms/cuDNN deterministic off; E4 both on. Training uses CUDA bfloat16 autocast. Dependency ranges in installation files support portable inspection, not guaranteed bitwise reproduction.

Model definitions and scientific engine functions were retained from the current R225 implementation. Imports and private path bindings were adapted for a standalone package. The command-line wrapper and public synthetic example are packaging additions; they were not used to obtain manuscript metrics. No research retraining was performed for this release. Existing raw-GIS preparation and original run management remain private; the input contract documents their required outputs. The example does not verify the full original data pipeline or establish paper-result reproducibility without restricted data.
