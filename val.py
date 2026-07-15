"""
Evaluate the trained model on the held-out TEST split and print metrics.

Usage:
    python val.py
"""

from ultralytics import YOLO

BEST_WEIGHTS = "runs/detect/runs_bib/yolo11n_bib_v1/weights/best.pt"
DATA_YAML = "dataset/data.yaml"


def main():
    model = YOLO(BEST_WEIGHTS)

    metrics = model.val(
        data=DATA_YAML,
        split="test",     # evaluate on test set, not val (val was already used during training)
        imgsz=512,
        batch=8,
        device=0,
        plots=True,        # saves confusion matrix, PR curve, etc. to runs_bib/.../val
    )

    print("\n--- Test set metrics ---")
    print("mAP50-95:", metrics.box.map)
    print("mAP50   :", metrics.box.map50)
    print("mAP75   :", metrics.box.map75)
    print("Precision (mean):", metrics.box.mp)
    print("Recall (mean)   :", metrics.box.mr)

    # Per-class breakdown (useful if you have multiple classes e.g. "bib", "bib-bike")
    print("\nPer-class mAP50-95:")
    for i, cls_map in enumerate(metrics.box.maps):
        cls_name = model.names[i]
        print(f"  {cls_name}: {cls_map:.4f}")


if __name__ == "__main__":
    main()