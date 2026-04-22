# LPSGM Grad-CAM

Grad-CAM + Guided Backpropagation pipeline for visualizing how LPSGM
attributes its sleep-stage predictions to specific PSG channels and time
regions. The default configuration targets the MASS-SS1 / MASS-SS3 public
datasets.

## Module Overview

| File                 | Purpose                                                                   |
|----------------------|---------------------------------------------------------------------------|
| `wrapper.py`         | `LPSGMInferenceWrapper` — single-arg `forward(x)` adapter over `model/`   |
| `channel_maps.py`    | MASS EDF channel mapping, canonical channel order, sleep-stage labels     |
| `utils.py`           | EDF I/O, signal preprocessing, MASS subject discovery and annotation parsing |
| `preparer.py`        | Shared EDF loading, EDF+ annotation alignment, and sliding-window construction |
| `save_raw.py`        | Stage 1: per-subject raw signals + LPSGM predictions + expert annotations |
| `custom_gradcam.py`  | Stage 2: dual-branch Grad-CAM at the last Conv1d of each Epoch Encoder    |
| `guided_backprop.py` | Stage 3: Guided Backpropagation saliency via ReLU replacement             |
| `render.py`          | Stage 4: per-epoch PNG visualizations (branch sum + envelope fusion)      |
| `pipeline.py`        | End-to-end CLI driver with stage selection                                |

## Running the Pipeline

From the repository root:

```bash
python -m gradcam.pipeline \
    --src-root <path_to_mass_edf_root> \
    --weights weights/ched32_seqed64_ch9_seql20_block4.pth \
    --output-root gradcam_output \
    --stages save_raw,gradcam,guided,render \
    --seq-len 20 --batch-size 16
```

`--src-root` must contain MASS-style EDF pairs: for each subject, both
`{sub_id} PSG.edf` (signals) and `{sub_id} Base.edf` (EDF+ annotations) are
required in the same directory.

## Stages

- **save_raw** — loads each EDF pair, aligns the signals against the EDF+
  annotations at 30-second resolution, runs LPSGM inference with
  `time_step=0` of each sliding window, and writes `raw_signal.npy`,
  `annotation.npy`, `prediction.npy`, `channels.txt` per subject to
  `{output-root}/raw/`.
- **gradcam** — computes dual-branch Grad-CAM at `encoder_branch{1,2}[11]`
  and writes `gradcam_branch1.npy` of shape `(num_windows, cn, 63)` and
  `gradcam_branch2.npy` of shape `(num_windows, cn, 13)` to
  `{output-root}/gradcam/`.
- **guided** — runs Guided Backpropagation at input resolution via in-place
  ReLU replacement and writes `guided_backpropagation.npy` of shape
  `(num_windows, cn, 3000)` to `{output-root}/guided/`.
- **render** — per-epoch PNG visualizations to `{output-root}/figures/`. For
  each epoch, the two-branch-summed Grad-CAM is rendered as a colormap
  behind each channel's raw waveform. This stage is CPU-only and uses a
  multiprocessing pool; it can be expensive for long recordings.

## Grad-CAM Aggregation Formula

For each 30-second epoch (the first epoch of each sliding window):

1. Compute branch-1 Grad-CAM at `encoder_branch1[11]` → `(cn, 63)`.
2. Compute branch-2 Grad-CAM at `encoder_branch2[11]` → `(cn, 13)`.
3. Linearly interpolate both along the time axis to 3000 samples.
4. **Element-wise sum** across branches → `gradcam_fused` of shape `(cn, 3000)`.
5. Compute the Hilbert envelope of the Guided Backpropagation output,
   Gaussian-smooth it (σ=10), and resample to 3000 samples.
6. Multiply: `guided_gradcam = gradcam_fused * envelope`.
7. Min-max normalize both maps to `[0, 1]`.

`render.py` uses `gradcam_fused` (step 4) as the per-sample background color
behind the raw waveform. `fuse_maps()` in `render.py` also exposes
`guided_gradcam` (step 6) for callers that require per-sample guided
attribution.
