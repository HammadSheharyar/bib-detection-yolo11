(base) PS E:\bib-detection-yolo11n> conda info --envs

# conda environments:
#
base                 * C:\Users\Pc\miniconda3
mmpose_gpu             C:\Users\Pc\miniconda3\envs\mmpose_gpu
yolo_gemma             C:\Users\Pc\miniconda3\envs\yolo_gemma

(base) PS E:\bib-detection-yolo11n> conda activate yolo_gemma
(yolo_gemma) PS E:\bib-detection-yolo11n> 

pip install ultralytics


  Epoch    GPU_mem   box_loss   cls_loss   dfl_loss  Instances       Size
      40/40      1.41G     0.9727     0.5226     0.9815         14        512: 100% ━━━━━━━━━━━━ 481/481 1.0it/s 7:44
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 29/29 1.5it/s 19.9s
                   all        453        982      0.901      0.888      0.927      0.599

40 epochs completed in 5.424 hours.
Optimizer stripped from E:\bib-detection-yolo11n\runs\detect\runs_bib\yolo11n_bib_v1\weights\last.pt, 5.5MB
Optimizer stripped from E:\bib-detection-yolo11n\runs\detect\runs_bib\yolo11n_bib_v1\weights\best.pt, 5.5MB

Validating E:\bib-detection-yolo11n\runs\detect\runs_bib\yolo11n_bib_v1\weights\best.pt...
Ultralytics 8.4.95  Python-3.11.15 torch-2.5.1+cu121 CUDA:0 (Quadro M1200, 4096MiB)
YOLO11n summary (fused): 101 layers, 2,582,347 parameters, 0 gradients, 6.3 GFLOPs
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 29/29 1.5it/s 19.8s
                   all        453        982      0.901      0.895      0.927      0.601
Speed: 0.6ms preprocess, 17.9ms inference, 0.0ms loss, 6.5ms postprocess per image


val: Scanning E:\bib-detection-yolo11n\dataset\test\labels... 226 images, 7 backgrounds, 0 corrupt: 100% ━━━━━━━━━━━━ 226/226 19.2it/s 11.8s
val: New cache created: E:\bib-detection-yolo11n\dataset\test\labels.cache
WARNING Box and segment counts should be equal, but got len(segments) = 10, len(boxes) = 413. To resolve this only boxes will be used and all segmentswill be removed. To avoid this please supply either a detect or segment dataset, not a detect-segment mixed dataset.
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100% ━━━━━━━━━━━━ 29/29 1.6it/s 17.7s
                   all        226        413      0.923      0.923      0.955      0.688
Speed: 7.4ms preprocess, 23.6ms inference, 0.0ms loss, 11.5ms postprocess per image
Results saved to E:\bib-detection-yolo11n\runs\detect\val

--- Test set metrics ---
mAP50-95: 0.6879605420911451
mAP50   : 0.954897004087765
mAP75   : 0.8141030525842446
Precision (mean): 0.9228464453892762
Recall (mean)   : 0.9225181598062954

Per-class mAP50-95:
  bib-number: 0.6880