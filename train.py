"""
Train YOLO11n on bib detection dataset.
Tuned for a 4GB VRAM GPU (Quadro M1200 / Maxwell arch).

Usage:
    python train.py
"""

from ultralytics import YOLO
import torch
# ********** ADDED **********
import time
import datetime


def main():
    # ********** ADDED: start time & log **********
    start_time = time.time()
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training started.")

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("VRAM (GB):", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2))

    model = YOLO("yolo11n.pt")

    results = model.train(
        # ---- CHANGE THIS to your actual path ----
        data="dataset/data.yaml",

        # Training duration
        epochs=40,
        patience=15,          # stop if no val improvement for 15 epochs

        # Image size / batch — sized for 4GB VRAM
        imgsz=512,             # 512 is risky on 4GB with mosaic on; 416 is safer
        batch=8,                # start here. If no OOM, you can try 12 or 16.
        # batch=-1,             # alternative: let ultralytics auto-pick batch size
                                 # based on ~60% free VRAM (safer than guessing)

        # Optimization
        optimizer="auto",
        cos_lr=True,
        lr0=0.01,
        lrf=0.01,

        # Augmentation — tuned down for small/text-like objects (bib numbers)
        mosaic=1.0,             # keep, good for varied backgrounds/crowds
        mixup=0.0,              # off — blending images hurts small text objects
        copy_paste=0.0,         # off for v1 — revisit only if you need more data variety
        degrees=5.0,
        shear=2.0,
        perspective=0.0005,     # small — helps with the bicycle handlebar angle variety
        flipud=0.0,             # correct — people/bikes are never upside down
        fliplr=0.5,             # fine — detection doesn't care about mirrored digits
        hsv_h=0.015,
        hsv_s=0.5,
        hsv_v=0.3,

        # System — tuned for laptop CPU + 4GB GPU
        device=0,
        workers=4,              # 8 is too many for most laptops, also crashes on Windows sometimes
        cache=False,            # "ram" cache can OOM your system RAM; leave off unless you have 16GB+ free
        amp=True,                # saves VRAM (matters more than speed on Maxwell)

        # Bookkeeping
        project="runs_bib",
        name="yolo11n_bib_v1",
        exist_ok=True,
        save=True,
        save_period=5,           # checkpoint every 5 epochs, in case training gets interrupted
        plots=True,
        val=True,
    )

    print("Best weights saved at:", results.save_dir / "weights" / "best.pt")

    # ********** ADDED: end time & log **********
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"[{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Training finished.")
    print(f"Total training time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")

    # ********** ADDED: optional – log summary to a file **********
    log_path = results.save_dir / "training_summary.txt"
    with open(log_path, "w") as f:
        f.write(f"Training started: {datetime.datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Training ended:   {datetime.datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total elapsed:    {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)\n")
        f.write(f"Best weights:     {results.save_dir / 'weights' / 'best.pt'}\n")
    print(f"Summary logged to: {log_path}")


if __name__ == "__main__":
    # The __main__ guard is required on Windows when workers > 0
    main()