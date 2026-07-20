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

# Bib Number Detection using YOLO26n

## Overview

This project trains a **YOLO26n (Ultralytics YOLO)** object detection model to detect **race bib numbers** in images. The model is fine-tuned on a custom dataset containing labeled bib number bounding boxes and achieves high detection accuracy while maintaining fast inference speed.

---

## Model Information

| Property | Value |
|----------|-------|
| Model | YOLO26n |
| Framework | Ultralytics YOLO v8.4.102 |
| Python | 3.12.13 |
| PyTorch | 2.10.0 + CUDA 12.8 |
| GPUs | 2 × Tesla T4 |
| Image Size | 512 × 512 |
| Task | Object Detection |
| Classes | 1 (Bib Number) |

---

# Dataset

The dataset consists of annotated race images with bib numbers labeled using YOLO format.

```
dataset/
│
├── train/
│   ├── images/
│   └── labels/
│
├── valid/
│   ├── images/
│   └── labels/
│
└── data.yaml
```

### Dataset Statistics

| Split | Images |
|--------|--------|
| Train | 4014 |
| Validation | 502 |
| Total | 4516 |

### Classes

```
0 : Bib Number
```

---

# Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 100 |
| Batch Size | 32 |
| Image Size | 512 |
| Optimizer | AdamW (Auto Selected) |
| Learning Rate | 0.002 |
| Weight Decay | 0.0005 |
| Mixed Precision | Enabled (AMP) |
| Multi-GPU | Yes (2 × Tesla T4) |
| Early Stopping | Patience = 20 |

Training command:

```python
from ultralytics import YOLO

model = YOLO("yolo26n.pt")

model.train(
    data="data.yaml",
    epochs=100,
    imgsz=512,
    batch=32,
    device=[0,1],
    name="yolo26n_bib"
)
```

---

# Training Results

## Final Validation Metrics

| Metric | Score |
|---------|-------|
| Precision | **0.903** |
| Recall | **0.897** |
| mAP@50 | **0.937** |
| mAP@50-95 | **0.596** |

These results indicate excellent detection accuracy with strong precision and recall on the validation dataset.

---

# Performance

| Metric | Time |
|---------|------|
| Preprocess | 0.1 ms |
| Inference | 1.2 ms |
| Postprocess | 0.8 ms |

Average inference speed per image:

**≈ 2.1 ms/image**

---

# Model Size

| File | Size |
|------|------|
| best.pt | 5.4 MB |
| last.pt | 5.4 MB |

Model summary:

- Parameters: **2,375,031**
- GFLOPs: **5.2**

---

# Results Directory

```
runs/
└── yolo26n_bib/
    ├── weights/
    │   ├── best.pt
    │   └── last.pt
    │
    ├── labels.jpg
    ├── results.png
    ├── results.csv
    └── args.yaml
```

---

# Inference

Load the trained model:

```python
from ultralytics import YOLO

model = YOLO("best.pt")
```

Run inference on an image:

```python
results = model.predict(
    source="image.jpg",
    imgsz=512,
    conf=0.25,
    save=True
)
```

Run inference on a folder:

```python
model.predict(
    source="images/",
    save=True
)
```

Run inference on a video:

```python
model.predict(
    source="video.mp4",
    save=True
)
```

---

# Export Model

Export to ONNX:

```python
model.export(format="onnx")
```

Export to TensorRT:

```python
model.export(format="engine")
```

Export to TorchScript:

```python
model.export(format="torchscript")
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/your-username/bib-number-detection.git
cd bib-number-detection
```

Install dependencies:

```bash
pip install ultralytics
```

Verify installation:

```bash
yolo checks
```

---

# Notes

- The training log reported a warning indicating that a few segmentation annotations were present in an otherwise detection-only dataset. During training, these segmentation labels were ignored, and only bounding boxes were used.
- Automatic Mixed Precision (AMP) was enabled to improve training efficiency.
- The optimizer was automatically selected by Ultralytics as **AdamW**.
- Mosaic augmentation was automatically disabled during the final training epochs for improved convergence.

---

# Future Improvements

- Train larger YOLO variants (YOLO26s, YOLO26m, YOLO26l)
- Increase image resolution (640 or 768)
- Add more diverse race environments
- Perform hyperparameter tuning
- Deploy using TensorRT for real-time inference
- Integrate OCR for automatic bib number recognition after detection

---

# Citation

If you use this project in your research or applications, please cite the Ultralytics YOLO framework:

```bibtex
@software{ultralytics,
  author = {Ultralytics},
  title = {Ultralytics YOLO},
  year = {2025},
  url = {https://github.com/ultralytics/ultralytics}
}
```

---

## License

This project is intended for educational and research purposes. Please ensure that any dataset used complies with its respective license and usage terms.
