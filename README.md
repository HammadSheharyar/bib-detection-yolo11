# Bib Number Detection using YOLO11n

A lightweight bib number detector trained using **YOLO11n (Ultralytics)** for athlete bib localization.

---

## Environment

### Hardware

- **GPU:** NVIDIA Quadro M1200 (4 GB VRAM)

### Software

- Python 3.11
- Conda
- PyTorch 2.5.1 + CUDA 12.1
- Ultralytics 8.4.95

---

## Setup

### Create / Activate Conda Environment

```bash
conda activate yolo_gemma
```

Verify available environments:

```bash
conda info --envs
```

Example:

```text
base
mmpose_gpu
yolo_gemma
```

---

### Install Dependencies

```bash
pip install ultralytics
```

Verify installation:

```bash
yolo version
```

---

## Training

Example training command:

```bash
yolo detect train \
    model=yolo11n.pt \
    data=dataset/data.yaml \
    imgsz=512 \
    epochs=40 \
    batch=16 \
    name=yolo11n_bib_v1
```

Training image size:

```
512 × 512
```

---

## Training Results

Training completed successfully.

### Validation Metrics

| Metric | Value |
|---------|--------|
| Precision | **0.901** |
| Recall | **0.895** |
| mAP@50 | **0.927** |
| mAP@50-95 | **0.601** |

Training time:

```
40 epochs
≈ 5.42 hours
```

Best weights:

```
runs/detect/runs_bib/yolo11n_bib_v1/weights/best.pt
```

---

## Test Set Results

| Metric | Value |
|---------|--------|
| Precision | **0.9228** |
| Recall | **0.9225** |
| mAP@50 | **0.9549** |
| mAP@75 | **0.8141** |
| mAP@50-95 | **0.6880** |

Per-class results:

| Class | mAP@50-95 |
|---------|-----------|
| bib-number | **0.6880** |

---

## Inference Speed

Using NVIDIA Quadro M1200 (4 GB)

| Stage | Time / Image |
|---------|--------------|
| Preprocess | 7.4 ms |
| Inference | 23.6 ms |
| Postprocess | 11.5 ms |

---

# TensorRT Export Notes

## Important

The model was trained with

```
imgsz = 512
```

When exporting to TensorRT, **always export using the same image size**.

Example:

```bash
yolo export \
    model=best.pt \
    format=engine \
    imgsz=512
```

---

## Why?

TensorRT engines are built for a fixed input resolution.

If exported using the default size (640), detection accuracy will degrade because the model was trained using 512×512 images.

---

## Prediction Limitation

Unlike PyTorch models,

```python
model.predict(imgsz=512)
```

**does not work for TensorRT (.engine) models.**

The engine input size is fixed during export.

Therefore:

- Export at the correct image size.
- Do **not** rely on changing `imgsz` during inference.

---

# Dataset Warning

During validation the following warning appeared:

```
WARNING:
Box and segment counts should be equal,
but got len(segments)=10,
len(boxes)=413
```

This indicates that the dataset contains a small number of segmentation annotations mixed with bounding-box annotations.

Ultralytics automatically ignores the segmentation polygons and trains using only bounding boxes.

Since only 10 segmentation annotations existed, this had negligible impact on training.

---

# Prediction Issue Encountered

A significant issue was observed during deployment.

The detector performs well on **full-scene images**, but confidence drops noticeably when detecting bib numbers inside **cropped person images**.

Re-exporting the model and changing TensorRT precision (FP16/FP32) did **not** solve the issue.

Therefore, the deployment format is **not** the root cause.

The issue originates from the training data distribution.

---

# Root Cause

The model was trained primarily on full-body or full-scene images.

During inference it is often given tightly cropped person images.

This creates a domain shift:

- Training input distribution ≠ Inference input distribution

As a result, confidence scores decrease even though the bib remains visible.

---

# Recommended Solution

The correct solution is **not** to modify TensorRT export settings.

Instead:

- Retrain using person-cropped images.
- Include both full-scene and cropped-person samples.
- Match the inference pipeline used during deployment.

Training on crop-style data will significantly improve confidence and robustness.

---

# Project Structure

```
dataset/
│
├── train/
├── valid/
├── test/
└── data.yaml

runs/
└── detect/
    ├── runs_bib/
    │   └── yolo11n_bib_v1/
    │       ├── weights/
    │       │   ├── best.pt
    │       │   └── last.pt
    │       └── results.png
```

---

# Summary

- ✅ YOLO11n
- ✅ NVIDIA Quadro M1200 (4 GB)
- ✅ Training image size: **512×512**
- ✅ Test mAP@50: **95.49%**
- ✅ Test mAP@50-95: **68.80%**
- ✅ TensorRT deployment supported
- ⚠️ Always export TensorRT using the same image size used during training.
- ⚠️ For cropped-person inference, retraining with crop-style data is recommended.
