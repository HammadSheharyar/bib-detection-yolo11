from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO


# ============================================================
# CONFIG — switch approach here
# ============================================================
APPROACH = "two_stage"   # "direct" = bib model on full image
                      # "two_stage" = person detect -> torso crop -> bib model

# Bib detection model (your trained weights)
BIB_WEIGHTS = "yolo11n-new/best.pt"
#BIB_WEIGHTS = "bibModelBest_server2.pt"


# Person detection model (pretrained COCO YOLO) — only used in two_stage
PERSON_WEIGHTS = "yolo11x.pt"   # auto-downloads on first run

SOURCE = "img (3).jpg"           # image path, folder, or video

# Inference params
IMGSZ = 640
CONF = 0.2
IOU = 0.45
DEVICE = 0                  # 0 = first GPU, "cpu" for CPU

# Two-stage tuning
PERSON_CONF = 0.35          # lower = more persons detected (may include FP)
TORSO_TOP = 0.15            # crop top fraction of person bbox (skip head)
TORSO_BOTTOM = 0.65         # crop bottom fraction (skip legs)
TORSO_PAD = 0.05            # extra padding around torso (fraction of bbox)


# ============================================================
# Helpers
# ============================================================
def torso_crop(img, xyxy, top=TORSO_TOP, bottom=TORSO_BOTTOM, pad=TORSO_PAD):
    """Return a torso crop of a person bbox and its coords in the original image."""
    x1, y1, x2, y2 = map(int, xyxy)
    h = y2 - y1
    w = x2 - x1

    # torso vertical slice
    ty1 = int(y1 + top * h)
    ty2 = int(y1 + bottom * h)

    # horizontal padding
    px = int(pad * w)
    tx1 = max(0, x1 - px)
    tx2 = min(img.shape[1], x2 + px)

    # clamp
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
    """Load image(s). Returns list of (path, ndarray)."""
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
# Approach 1: direct bib detection on full image
# ============================================================
def run_direct(bib_model, items):
    print("\n=== Approach: DIRECT (bib model on full image) ===")
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

        out = f"out_direct_{Path(path).stem}.jpg"
        cv2.imwrite(out, vis)
        print(f"{path} -> {len(r.boxes)} bib(es)  | saved {out}")


# ============================================================
# Approach 2: person -> torso -> bib model
# ============================================================
def run_two_stage(person_model, bib_model, items):
    print("\n=== Approach: TWO-STAGE (person -> torso -> bib) ===")
    for path, img in items:
        # 1) detect persons
        pres = person_model.predict(
            source=img, imgsz=IMGSZ, conf=PERSON_CONF, iou=IOU,
            classes=[0],          # COCO class 0 = person
            device=DEVICE, verbose=False,
        )[0]

        vis = img.copy()
        total_bibs = 0

        for pbox in pres.boxes:
            pxyxy = pbox.xyxy[0].cpu().numpy()

            # draw person bbox (blue)
            draw_box(vis, pxyxy, label="person", color=(255, 128, 0), thickness=1)

            # 2) crop torso
            crop, (tx1, ty1, tx2, ty2) = torso_crop(img, pxyxy)
            if crop.size == 0:
                continue

            # draw torso region (yellow, dashed feel via thin line)
            draw_box(vis, (tx1, ty1, tx2, ty2), color=(0, 255, 255), thickness=1)

            # 3) bib detection on crop
            bres = bib_model.predict(
                source=crop, imgsz=IMGSZ, conf=CONF, iou=IOU,
                device=DEVICE, verbose=False,
            )[0]

            for bbox in bres.boxes:
                bx1, by1, bx2, by2 = bbox.xyxy[0].cpu().numpy()
                conf = float(bbox.conf[0])

                # map back to original image coords
                gx1 = tx1 + bx1
                gy1 = ty1 + by1
                gx2 = tx1 + bx2
                gy2 = ty1 + by2

                draw_box(vis, (gx1, gy1, gx2, gy2),
                         label=f"bib {conf:.2f}", color=(0, 255, 0), thickness=2)
                total_bibs += 1

        out = f"out_twostage_{Path(path).stem}.jpg"
        cv2.imwrite(out, vis)
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

    if APPROACH == "direct":
        run_direct(bib_model, items)
    elif APPROACH == "two_stage":
        person_model = YOLO(PERSON_WEIGHTS)
        run_two_stage(person_model, bib_model, items)
    else:
        raise ValueError(f"Unknown APPROACH: {APPROACH}")


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



