import os, sys
sys.path.insert(0, "/home/gaya7/dsc3032-gaya-shared/group7/official_yoloe")
from ultralytics import YOLOE

# Own Code
# Evaluates the full fine-tuning checkpoint and prints box/mask AP for Table 4 comparison.

model = YOLOE("/home/gaya7/dsc3032-gaya-shared/group7/runs/table4_yoloe_v8s_ft_fullcoco_ep160_b16/weights/best.pt")
metrics = model.val(
    data="/home/gaya7/dsc3032-gaya-shared/group7/datasets/coco_full_yolo_seg/coco_subset_seg.yaml",
    imgsz=640, batch=16, device=1, split="val", task="segment"
)
print(f"\nAPb={metrics.box.map*100:.2f}, APb50={metrics.box.map50*100:.2f}")
print(f"APm={metrics.seg.map*100:.2f}, APm50={metrics.seg.map50*100:.2f}")
