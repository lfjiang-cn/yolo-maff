# YOLO-MAFF

YOLO-MAFF is a project-specific fork of Ultralytics YOLOv8 for small object detection, especially traffic signs.

This repository keeps the Ultralytics training/inference pipeline while adding two custom ideas:

- AMC-CAM in the backbone (`C2f_AMC_CAM`)
- Adaptive Feature Fusion in the detection head (`Detect_AFSFF`, with SFFM)

## 1. Project Highlights

### 1.1 AMC-CAM and C2f-AMC-CAM

- File: `ultralytics/nn/Addmodules/C2f_AMC_CAM.py`
- Purpose:
  - Add multi-scale channel attention to Bottleneck/C2f.
  - Fuse global channel context and local multi-scale context.

### 1.2 Adaptive Feature Fusion + SFFM

- File: `ultralytics/nn/Addmodules/AdaptiveFusionHead.py`
- Purpose:
  - Align P2/P3/P4/P5 features to each target scale.
  - Learn spatial fusion weights with SFFM (softmax normalized).
  - Produce fused features `F2/F3/F4/F5` for final detection.

### 1.3 Model Config Used in This Repo

- File: `ultralytics/cfg/models/v8/yolo-maff.yaml`
- Backbone uses `C2f_AMC_CAM`.
- Detection head uses `Detect_AFSFF`.



## 2. Environment Setup

Recommended environment (example):

```bash
conda create -n envzpd python=3.9 -y
conda activate envzpd
```

Install dependencies from this repo:

```bash
pip install -e .
```

Check GPU/torch quickly:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```



## 3. Training

### 3.1 Use Provided Script

```bash
python train.py
```


### 3.2 CLI Alternative

```bash
yolo detect train \
  model=ultralytics/cfg/models/v8/yolo-maff.yaml \
  data=ultralytics/cfg/datasets/tt100k.yaml \
  epochs=300 imgsz=640 batch=4 device=0 workers=0
```

## 4. Validation and Inference

### 4.1 Validation

```bash
yolo detect val \
  model=runs/train/v8s/weights/best.pt \
  data=ultralytics/cfg/datasets/tt100k.yaml \
  imgsz=640 device=0
```

### 4.2 Inference

```bash
yolo detect predict \
  model=runs/train/v8s/weights/best.pt \
  source=path/to/images \
  imgsz=640 device=0
```

## 5. Notes

- When `amp=True`, Ultralytics performs an AMP check before training.
- This repo is configured to avoid training interruption if the AMP demo image is missing.
- You may still set `amp=False` in `train.py` if you want to skip AMP behavior entirely.

## 6. Reproducibility Tips

- Keep `seed=0` and `deterministic=True` for more stable comparisons.
- Record:
  - commit hash
  - dataset split
  - model config (`yolo-maff.yaml`)
  - training arguments

## 7. Acknowledgement

This project is built on top of:

- Ultralytics YOLOv8: https://github.com/ultralytics/ultralytics

## 8. License

This repository follows the same license declared in the project:

- AGPL-3.0

Please check `LICENSE` for details.
