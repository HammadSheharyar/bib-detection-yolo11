# """
# Run inference with the trained model on new images/videos and save annotated output.

# Usage:
#     python predict.py
# """

# from ultralytics import YOLO

# BEST_WEIGHTS = "runs/detect/runs_bib/yolo11n_bib_v1/weights/best.pt"
# SOURCE = "peop7.png"


# def main():
#     model = YOLO(BEST_WEIGHTS)

#     results = model.predict(
#         source=SOURCE,
#         imgsz=512,
#         conf=0.2,       # detection confidence threshold — lower to catch more, raise to reduce false positives
#         iou=0.45,
#         device=0,
#         save=True,        # saves annotated images/video to runs_bib/detect/predict*
#         save_txt=False,   # set True if you want YOLO-format label files out
#         stream=False,
#     )

#     for r in results:
#         print(r.path, "->", len(r.boxes), "bib(es) detected")


# if __name__ == "__main__":
#     main()

from ultralytics import YOLO

# BEST_WEIGHTS = "runs/detect/runs_bib/yolo11n_bib_v1/weights/best.pt"
# BEST_WEIGHTS = "bestyolo26n.pt"
SOURCE = "test_images_2"


BEST_WEIGHTS = "yolo11n-new/best.pt"

# SOURCE = "dataset/test/images"


def main():
    model = YOLO(BEST_WEIGHTS)

    results = model.predict(
        source=SOURCE,
        imgsz=640,
        conf=0.2,
        iou=0.45,
        device=0,
        save=True,
        save_txt=False,
        rect = False,
        stream=False,
        show_labels=False,   # Hide class name ("bib")
        show_conf=True,      # Show confidence only
    )

    for r in results:
        print(r.path, "->", len(r.boxes), "bib(es) detected")


if __name__ == "__main__":
    main()