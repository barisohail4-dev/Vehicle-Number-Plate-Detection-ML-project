# Automatic License Plate Recognition using YOLOv8 and EasyOCR

This project builds a complete end-to-end pipeline for automatic license plate recognition using YOLOv8 for detection and EasyOCR for optical character recognition. It is designed to work with Pascal VOC XML annotations, convert them into YOLO labels, train a detector, run inference on new images, crop detected plates, and recognize plate text.

## Features
- Convert Pascal VOC XML annotations to YOLO format
- Split data into train, validation, and test sets
- Create dataset.yaml automatically
- Train a YOLOv8n detector
- Save best.pt and last.pt checkpoints
- Detect plates from new images
- Crop detected plate regions
- Read plate text using EasyOCR
- Save predictions and recognized numbers
- Provide a simple local web app entry point

## Project Structure
```text
Automatic-License-Plate-Recognition/
├── dataset/
│   ├── raw/
│   │   ├── images/
│   │   └── annotations/
│   ├── yolo/
│   │   ├── images/
│   │   ├── labels/
│   │   └── dataset.yaml
│   └── README.md
├── notebooks/
├── src/
├── models/
├── outputs/
├── runs/
├── app/
├── requirements.txt
├── README.md
├── main.py
```

## Installation
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

## Dataset Preparation
```bash
python main.py --prepare
```

## Training
```bash
python main.py --train --epochs 20 --imgsz 640
```

## Inference
```bash
python main.py --infer path/to/image.jpg
```

## Web App
```bash
python app/app.py
```
Then open http://127.0.0.1:5000/ in your browser.

## Expected Results
- Detection model weights stored in models/best.pt and models/last.pt
- Prediction images saved in outputs/predictions
- Cropped plates saved in outputs/cropped_plates
- Recognized values saved in outputs/results.csv

## Notes
- This project targets a professional ML pipeline structure for academic and demo purposes.
- Accuracy depends on the image quality and annotation coverage of the dataset.
- For best performance, use a larger dataset and train for more epochs.
