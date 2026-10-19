"""Download the AI model weights into backend/models so the app can run without internet access.

  models/yolov8n.pt     - YOLOv8n furniture detector (~6 MB)
  models/mask2former-small/ - Mask2Former (Swin-Small) ADE20K room segmenter (~275 MB)

Usage:  .venv\\Scripts\\python.exe scripts\\download_models.py
"""
import os
import shutil
import sys

# The server runs the app with HF_HUB_OFFLINE=1; this script is the one place that must go online.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
sys.path.insert(0, str(ROOT))


def main():
    MODELS.mkdir(exist_ok=True)

    yolo = MODELS / "yolov8n.pt"
    if not yolo.exists():
        from ultralytics import YOLO

        YOLO("yolov8n.pt")  # downloads into the current directory
        shutil.move("yolov8n.pt", yolo)
    print(f"YOLO weights:      {yolo} ({yolo.stat().st_size / 1e6:.1f} MB)")

    from app.services.segmentation import LOCAL_DIR, MODEL_ID
    from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

    if not (LOCAL_DIR / "config.json").exists():
        AutoImageProcessor.from_pretrained(MODEL_ID).save_pretrained(LOCAL_DIR)
        Mask2FormerForUniversalSegmentation.from_pretrained(MODEL_ID).save_pretrained(LOCAL_DIR)
    size = sum(f.stat().st_size for f in LOCAL_DIR.rglob("*") if f.is_file()) / 1e6
    print(f"Segmenter weights: {LOCAL_DIR} ({size:.1f} MB)")


if __name__ == "__main__":
    main()
