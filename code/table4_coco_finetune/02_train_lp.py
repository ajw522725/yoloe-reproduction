import os, sys
sys.path.insert(0, "/home/gaya7/dsc3032-gaya-shared/group7/official_yoloe")
from ultralytics import YOLOE

os.environ["PYTHONHASHSEED"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

model = YOLOE("/home/gaya7/dsc3032-gaya-shared/group7/official_yoloe/pretrain/yoloe-v8s-seg.pt")

results = model.train(
    data="/home/gaya7/dsc3032-gaya-shared/group7/datasets/coco_full_yolo_seg/coco_subset_seg.yaml",
    task="segment",
    epochs=10,
    imgsz=640,
    batch=64,
    device=1,
    workers=4,
    optimizer="AdamW",
    lr0=0.001,
    weight_decay=0.025,
    freeze=22,
    project="/home/gaya7/dsc3032-gaya-shared/group7/runs",
    name="table4_yoloe_v8s_lp_fullcoco_ep10_b64",
    exist_ok=True,
    val=True,
    plots=True,
)
print(results)
