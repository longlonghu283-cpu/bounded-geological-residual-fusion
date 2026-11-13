# Prepared input contract

Private GIS products and scene-specific partitions are not distributed. The engines consume arrays already registered to a common projected grid. Preparation must use authorised inputs; no occurrence coordinates or georeferencing constants are embedded here.

| File | Contents |
| --- | --- |
| HSI_158_FLOAT32.npy | float32, (158,H,W), supplied radiometric scaling applied; eight all-zero source bands removed |
| COMMON_VALID.npy | Boolean (H,W), shared image validity |
| R225_LABEL.npy | Binary (H,W), pixel-centre distance to nearest known occurrence <=225 m |
| EVALUATION_SUPPORT.npy | Integer (H,W), >0 only in scored 150 m cores and 450–900 m reference-background annuli |
| MAP_UNIT_CODES.npy | Nonnegative integer (H,W), raw categorical codes, 0 missing |
| FAULT_DISTANCE_M.npy | float32 (H,W), distance to nearest mapped fault; zero is valid |
| GEO_VALID.npy | Boolean (H,W), valid geology coverage |
| FOLDk/MANIFEST.json | train_P, train_U and val_P lists of {y,x} 64-pixel window origins; positive_class_weight |
| FOLDk/NORMALIZATION.npz | mean and std, each (158,), deduplicated valid training-P pixels only |
| FOLDk/GEOLOGY_LOCK.json | raw_code_to_index, n_embeddings, fault_mean_m, fault_std_m; training-P only |
| FOLDk/DEV_SUPPORT.npz | sorted unique flattened-coordinate keys and corresponding binary labels |

The index convention is 0 for missing, 1 for valid but unseen map units, and 2 onward for categories observed in training. A per-fold category dictionary must not use validation attributes to fit its vocabulary. Continuous distance parameters and spectral statistics use the deduplicated training P footprint. Clamp standard deviations away from zero during cache preparation. Spectral values are clipped to [-5,5] after standardisation. Masks and category maps use nearest-neighbour sampling. Polygon categories are rasterised at pixel centres; conflicting overlaps are treated as missing. Fault distance is Euclidean in the projected metric grid. Original fault-property categories are not model inputs.

Training positives and DEV scoring labels differ deliberately. A 225 m training buffer is not an orebody boundary. Pixels in a P window outside that buffer are operational local background, not field-confirmed barren ground. U windows have no direct label supervision. DEV uses 150 m cores and background 450–900 m away; transition and exterior pixels are unscored. Where buffers overlap, a pixel inside any positive core remains positive. DEV keys are global row*width+column values; repeated predictions are averaged before scoring.

Four folds hold out northwest, northeast, southwest and southeast windows respectively. P, U and DEV footprints must be disjoint within a fold. Training and DEV can still share occurrence identities. Validate the frozen manifest before execution; do not regenerate partitions when claiming reproduction of an existing run. The public package deliberately excludes the original manifests because they reveal occurrence locations.

## Required files by stage

E3 reads the HSI, image-valid mask, R225 training-label raster and evaluation-support mask, plus `MANIFEST.json`, `NORMALIZATION.npz` and `DEV_SUPPORT.npz` for its fold. E3 does not read geological inputs.

Every E4 variant additionally reads `MAP_UNIT_CODES.npy`, `FAULT_DISTANCE_M.npy`, `GEO_VALID.npy` and the fold's `GEOLOGY_LOCK.json`. This includes `hsi`: the E4 data adapter still loads these arrays even when the model bypasses the geological branch. The `noise` runner generates its substitute inputs in the run directory; they are not an additional raw-data requirement.

For a four-fold E4 cache, the directory layout is:

```text
cache/
  HSI_158_FLOAT32.npy
  COMMON_VALID.npy
  R225_LABEL.npy
  EVALUATION_SUPPORT.npy
  MAP_UNIT_CODES.npy
  FAULT_DISTANCE_M.npy
  GEO_VALID.npy
  FOLD0/
    MANIFEST.json
    NORMALIZATION.npz
    GEOLOGY_LOCK.json
    DEV_SUPPORT.npz
  FOLD1/   (same four filenames, fold-specific contents)
  FOLD2/   (same four filenames, fold-specific contents)
  FOLD3/   (same four filenames, fold-specific contents)
```

These paths describe private prepared inputs; none of the cache files is distributed in the public repository.

## Exact metadata contract

`MANIFEST.json` is an object containing:

- `train_P`: a non-empty list of labelled training-window objects.
- `train_U`: a non-empty list of unlabelled training-window objects.
- `val_P`: a non-empty list of development-window objects.
- `positive_class_weight`: a finite positive number. The study estimates this from the training fold and caps it at 20 during preparation.

Each window object has integer `y` and `x` fields: zero-based top-left row and column. Require `0 <= y <= H-64` and `0 <= x <= W-64`. Coordinates are array indices, not projected metres. Values in the original manifests cannot be reconstructed merely from the public table of window counts.

`NORMALIZATION.npz` must have named arrays `mean` and `std`, each with shape `(158,)`. Both must be finite, and every standard deviation must be strictly positive. Derive them from deduplicated valid training-P pixels, before model-side clipping. The HSI file contains scaled spectral values; do not standardise it twice.

`GEOLOGY_LOCK.json` contains these required keys:

| Key | Required meaning |
| --- | --- |
| `raw_code_to_index` | JSON object whose keys are string representations of nonnegative integer map-unit codes and values are embedding indices |
| `n_embeddings` | Integer embedding-table size; every index is smaller than this value |
| `fault_mean_m` | Finite mean distance computed from valid geology in the deduplicated training-P footprint |
| `fault_std_m` | Finite, strictly positive training-only distance standard deviation |

Missing category index is 0; valid unseen categories use 1; observed training categories use indices from 2. All dictionary codes must fit the dense lookup range `0..MAP_UNIT_CODES.max()` used by the released adapter. The `noise` control requires `n_embeddings > 2`, since substitute categories are sampled from indices 2 onward. Do not fit the dictionary or continuous scaling parameters on the DEV footprint.

`DEV_SUPPORT.npz` must contain `keys` and `labels`:

- Both are one-dimensional arrays of equal nonzero length.
- `keys` should be `int64`, sorted, unique and within `[0, H*W)`.
- `labels` are binary 0/1 and must include both classes; metric functions require both.
- For each `val_P` window, select positions where `EVALUATION_SUPPORT > 0`, convert them to `row*W + column`, combine all windows, and deduplicate. This sorted union must equal `keys` exactly.
- `labels` must correspond to those keys in the same order. Positive labels refer to the 150 m cores; negative labels refer to the 450-900 m reference-background support.

`EVALUATION_SUPPORT` is a selection mask: any positive value selects a position. Its numeric values are **not** used as class labels by the evaluator; scoring labels come from `DEV_SUPPORT.labels`. Ensure all scored positions are image-valid. The transition zone and the exterior of the background annulus must be excluded.

## Array integrity checks before execution

All rasters share `(H, W)`, orientation, pixel origin and projected grid. The HSI has channels first, `(158, H, W)`. At least one complete 64 x 64 window must fit. Map-unit values are nonnegative integers; Boolean masks have consistent coverage.

Provide finite HSI values even in masked areas: the adapter multiplies by the validity mask, and `NaN * 0` remains NaN. Use finite placeholder values there. Fault distances are nonnegative, finite on valid geological coverage, and are in metres; zero distance remains a valid observation. Normalisation and geometric registration are input-preparation responsibilities, not operations inferred by the runner.

The engine checks some invariants during evaluation, but it is not a comprehensive cache validator. Confirm shape, finiteness, coverage, window separation, training-only fitting and key-label correspondence before starting a run.
