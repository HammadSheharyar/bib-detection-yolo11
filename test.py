from pathlib import Path
from datetime import datetime
import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIG
# ============================================================
APPROACH = "direct"   # "direct" or "two_stage"

BIB_WEIGHTS = "yolo11n-new/best.pt"
# BIB_WEIGHTS = "bibModelBest_server2.pt"

PERSON_WEIGHTS = "yolo11x.pt"

SOURCE = "img (3).jpg"

IMGSZ = 640
CONF = 0.2
IOU = 0.45
DEVICE = 0

# Two-stage tuning
PERSON_CONF = 0.35
TORSO_TOP = 0.15
TORSO_BOTTOM = 0.65
TORSO_PAD = 0.05

# Output root — same style as ultralytics
RUNS_ROOT = "runs"


# ============================================================
# Output folder (per run)
# ============================================================
def make_run_dir(approach: str) -> Path:
    """
    Create a unique run directory like:
        runs/detect/direct_20251013_143022/
    If a same-second folder exists, appends _2, _3, ...
    """
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = Path(RUNS_ROOT) / "detect" / f"{approach}_{stamp}"

    if base.exists():
        i = 2
        while (base.parent / f"{base.name}_{i}").exists():
            i += 1
        base = base.parent / f"{base.name}_{i}"

    base.mkdir(parents=True, exist_ok=False)
    return base


# ============================================================
# Helpers
# ============================================================
def torso_crop(img, xyxy, top=TORSO_TOP, bottom=TORSO_BOTTOM, pad=TORSO_PAD):
    x1, y1, x2, y2 = map(int, xyxy)
    h = y2 - y1
    w = x2 - x1

    ty1 = int(y1 + top * h)
    ty2 = int(y1 + bottom * h)

    px = int(pad * w)
    tx1 = max(0, x1 - px)
    tx2 = min(img.shape[1], x2 + px)

    ty1 = max(0, ty1)
    ty2 = min(img.shape[0], ty2)

    crop = img[ty1:ty2, tx1:tx2]
    return crop, (tx1, ty1, tx2, ty2)


def draw_box(img, xyxy, label=None, color=(0, 255, 0), thickness=2):
    x1, y1, x2, y2 = map(int, xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if label:
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


def load_source(source):
    p = Path(source)
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = sorted([f for f in p.iterdir() if f.suffix.lower() in exts])
    else:
        files = [p]

    items = []
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"[WARN] Could not read {f}")
            continue
        items.append((str(f), img))
    return items


# ============================================================
# Approach 1: direct
# ============================================================
def run_direct(bib_model, items, out_dir: Path):
    print(f"\n=== Approach: DIRECT ===\nSaving to: {out_dir}")
    for path, img in items:
        results = bib_model.predict(
            source=img, imgsz=IMGSZ, conf=CONF, iou=IOU,
            device=DEVICE, verbose=False,
        )
        r = results[0]
        vis = img.copy()
        for box in r.boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0])
            draw_box(vis, xyxy, label=f"{conf:.2f}", color=(0, 255, 0))

        out = out_dir / f"{Path(path).stem}.jpg"
        cv2.imwrite(str(out), vis)
        print(f"{path} -> {len(r.boxes)} bib(es)  | saved {out}")


# ============================================================
# Approach 2: person -> torso -> bib
# ============================================================
def run_two_stage(person_model, bib_model, items, out_dir: Path):
    print(f"\n=== Approach: TWO-STAGE ===\nSaving to: {out_dir}")

    # optional: also save intermediate torso crops for debugging
    crops_dir = out_dir / "torso_crops"
    crops_dir.mkdir(exist_ok=True)

    for path, img in items:
        pres = person_model.predict(
            source=img, imgsz=IMGSZ, conf=PERSON_CONF, iou=IOU,
            classes=[0], device=DEVICE, verbose=False,
        )[0]

        vis = img.copy()
        total_bibs = 0
        stem = Path(path).stem

        for i, pbox in enumerate(pres.boxes):
            pxyxy = pbox.xyxy[0].cpu().numpy()
            draw_box(vis, pxyxy, label="person", color=(255, 128, 0), thickness=1)

            crop, (tx1, ty1, tx2, ty2) = torso_crop(img, pxyxy)
            if crop.size == 0:
                continue

            draw_box(vis, (tx1, ty1, tx2, ty2), color=(0, 255, 255), thickness=1)

            # save torso crop (debug)
            cv2.imwrite(str(crops_dir / f"{stem}_person{i}.jpg"), crop)

            bres = bib_model.predict(
                source=crop, imgsz=IMGSZ, conf=CONF, iou=IOU,
                device=DEVICE, verbose=False,
            )[0]

            for bbox in bres.boxes:
                bx1, by1, bx2, by2 = bbox.xyxy[0].cpu().numpy()
                conf = float(bbox.conf[0])

                gx1 = tx1 + bx1
                gy1 = ty1 + by1
                gx2 = tx1 + bx2
                gy2 = ty1 + by2

                draw_box(vis, (gx1, gy1, gx2, gy2),
                         label=f"bib {conf:.2f}", color=(0, 255, 0), thickness=2)
                total_bibs += 1

        out = out_dir / f"{stem}.jpg"
        cv2.imwrite(str(out), vis)
        print(f"{path} -> {len(pres.boxes)} person(s), {total_bibs} bib(es)  | saved {out}")


# ============================================================
# Main
# ============================================================
def main():
    items = load_source(SOURCE)
    if not items:
        print("No images loaded.")
        return

    print(f"Loaded {len(items)} image(s) from '{SOURCE}'")
    bib_model = YOLO(BIB_WEIGHTS)

    out_dir = make_run_dir(APPROACH)

    if APPROACH == "direct":
        run_direct(bib_model, items, out_dir)
    elif APPROACH == "two_stage":
        person_model = YOLO(PERSON_WEIGHTS)
        run_two_stage(person_model, bib_model, items, out_dir)
    else:
        raise ValueError(f"Unknown APPROACH: {APPROACH}")

    print(f"\n✅ Done. Results in: {out_dir}")


if __name__ == "__main__":
    main()





















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

# from ultralytics import YOLO

# # BEST_WEIGHTS = "runs/detect/runs_bib/yolo11n_bib_v1/weights/best.pt"
# # BEST_WEIGHTS = "bestyolo26n.pt"
# SOURCE = "img (3).jpg"


# BEST_WEIGHTS = "yolo11n-new/best.pt"
# # BEST_WEIGHTS = "bibModelBest_server2.pt"


# # SOURCE = "dataset/test/images"


# def main():
#     model = YOLO(BEST_WEIGHTS)

#     results = model.predict(
#         source=SOURCE,
#         imgsz=640,
#         conf=0.2,
#         iou=0.45,
#         device=0,
#         save=True,
#         save_txt=False,
#         rect = False,
#         stream=False,
#         show_labels=False,   # Hide class name ("bib")
#         show_conf=True,      # Show confidence only
#     )

#     for r in results:
#         print(r.path, "->", len(r.boxes), "bib(es) detected")


# if __name__ == "__main__":
#     main()



