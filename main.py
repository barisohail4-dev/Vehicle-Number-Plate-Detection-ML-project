from __future__ import annotations

import argparse
from pathlib import Path

from src.config import DEFAULT_SAMPLE_IMAGE, ensure_directories
from src.data_preprocessing import prepare_dataset
from src.train import train_model
from src.inference import run_inference


def main() -> None:
    parser = argparse.ArgumentParser(description="Automatic License Plate Recognition")
    parser.add_argument("--prepare", action="store_true", help="Convert XML annotations to YOLO labels")
    parser.add_argument("--train", action="store_true", help="Train the YOLOv8 model")
    parser.add_argument("--infer", type=str, help="Path to an image for inference")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    ensure_directories()

    if args.prepare:
        prepare_dataset()
        print("Dataset preparation complete.")

    if args.train:
        result = train_model(epochs=args.epochs, imgsz=args.imgsz)
        print("Training complete:", result)

    if args.infer:
        image_path = Path(args.infer)
        result = run_inference(image_path)
        print("Inference complete:", result)

    if not any([args.prepare, args.train, args.infer]):
        image_path = DEFAULT_SAMPLE_IMAGE
        result = run_inference(image_path)
        print("Default inference complete:", result)


if __name__ == "__main__":
    main()
