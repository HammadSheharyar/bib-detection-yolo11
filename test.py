"""
Run inference with the trained model on new images/videos and save annotated output.

Usage:
    python predict.py
"""

from ultralytics import YOLO

BEST_WEIGHTS = "runs/detect/runs_bib/yolo11n_bib_v1/weights/best.pt"
SOURCE = "path/to/some/images_or_video_or_folder"


def main():
    model = YOLO(BEST_WEIGHTS)

    results = model.predict(
        source=SOURCE,
        imgsz=416,
        conf=0.25,       # detection confidence threshold — lower to catch more, raise to reduce false positives
        iou=0.45,
        device=0,
        save=True,        # saves annotated images/video to runs_bib/detect/predict*
        save_txt=False,   # set True if you want YOLO-format label files out
        stream=False,
    )

    for r in results:
        print(r.path, "->", len(r.boxes), "bib(es) detected")


if __name__ == "__main__":
    main()