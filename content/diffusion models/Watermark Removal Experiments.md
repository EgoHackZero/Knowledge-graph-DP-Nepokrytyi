# Watermark Removal Experiments

## Task

Remove attribution watermarks from Google Maps aerial imagery before using the dataset for diffusion model training.

**Watermark**: `Image Landsat / Copernicus` — small semi-transparent white text centered at the bottom edge of each image (512×512 px).

**Dataset**: `DP-Nepokrytyi-all-data_unprepare/filtered data/`
- `test/*` — folders with numeric names (`0000`, `0001`, ...)
- `train/*_drone` — drone imagery

---

## Approaches and Results

### Experiment 1 — EasyOCR Detection + Dynamic Mask

**Idea**: detect text with EasyOCR, build a mask from bounding boxes, inpaint with LaMa.

**Result**: failed. EasyOCR could not reliably detect small low-contrast Google Maps text — the majority of images were skipped (zero detections returned).

**Conclusion**: EasyOCR alone is not suitable for this watermark type.

---

### Experiment 2 — EasyOCR as Filter + Fixed Mask

**Idea**: use EasyOCR only as a presence check — if text detected, apply a fixed mask; otherwise skip the image.

**Initial fixed mask parameters**:
- `fixed_bottom_frac: 0.05` → mask height 26px out of 512 — too small
- `fixed_center_frac: 0.30` → width 154px — did not cover the full text

**Result**: some images were processed, but the watermark remained visible — the mask did not cover the actual text position (~60px from the bottom).

---

### Experiment 3 — Corrected Mask Coordinates

**Updated parameters**:
- `fixed_bottom_frac: 0.12` → 61px from the bottom (covers the full watermark with margin)
- `fixed_center_frac: 0.40` → 205px wide (center 40% of image width)

**Result**: watermark fully covered by the mask.

---

### Experiment 4 — LaMa Without Refiner (Direct Inference)

**Model**: `big-lama` (large mask inpainting), `pad_to_modulo=32`.

**Result**: inpainting worked without visible artifacts, but the texture in the masked region appeared slightly smoother than the surroundings.

---

### Experiment 5 — LaMa With Refiner

**Parameters**: `n_iters=15`, `max_scales=3`.

**Technical issues encountered**:
- `predict.py` names the output file after the **mask** (`img_mask.png`), not the image (`img.png`) — fixed in code
- PyTorch 2.6+ requires `weights_only=False` for LaMa checkpoints — fixed in `trainers/__init__.py`
- Having both `opencv-python` and `opencv-python-headless` installed caused a C++ crash — resolved by keeping only `opencv-python==4.8.1.78`
- `numpy>=2.0` is incompatible with albumentations 0.5.2 — pinned `numpy<2.0`

**Result**: inpainting quality improved — texture in the masked region matches the surroundings more naturally.

---

## Final Solution

**EasyOCR removed** — since the watermark is present on every image in the dataset and its position is fixed, detection is redundant and unreliable. Every image is processed with the fixed mask unconditionally.

**Pipeline**: fixed mask → LaMa with refiner.

**Run modes**:
```bash
# In-place (overwrites originals)
python remove_text.py --data_root "path/to/filtered data" --device cuda

# To a separate output folder
python remove_text.py --input_dir path/to/folder --device cuda
```

---

## Final Parameters (`config.yaml`)

```yaml
detection:
  fixed_bottom_frac: 0.12   # 61px for 512px images
  fixed_center_frac: 0.40   # center 40% of image width

refiner:
  enabled: true
  n_iters: 15
  max_scales: 3
```

---

## Result Assessment

Final images (`0000_text_removed`) were visually assessed as suitable for the dataset:
- Watermark fully removed
- No visible inpainting artifacts
- 512×512 resolution preserved
- Consistent aerial photography style suitable for diffusion model training
