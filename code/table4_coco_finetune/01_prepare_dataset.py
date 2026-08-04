# Own Code
# Converts COCO annotations into YOLO detection/segmentation labels and dataset YAML files.
# Added to prepare COCO data for YOLOE Table 4 fine-tuning.
import argparse
import json
import os
import random
import shutil
from pathlib import Path
from collections import defaultdict

import yaml
from tqdm import tqdm


def link_or_copy(src, dst, mode):
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if mode == "symlink":
        os.symlink(src, dst)
    else:
        shutil.copy2(src, dst)


def clean_name(name):
    return " ".join(str(name).replace("/", " ").split())


def build_class_mapping(categories):
    sorted_categories = sorted(categories, key=lambda x: x["id"])

    old_to_new = {}
    names = {}

    for new_id, cat in enumerate(sorted_categories):
        old_to_new[cat["id"]] = new_id
        names[new_id] = clean_name(cat["name"])

    return old_to_new, names


def coco_bbox_to_yolo_line(cls_id, bbox, img_w, img_h):
    # COCO bbox -> YOLO normalized bbox.
    x, y, w, h = bbox

    if w <= 0 or h <= 0:
        return None

    xc = (x + w / 2.0) / img_w
    yc = (y + h / 2.0) / img_h
    bw = w / img_w
    bh = h / img_h

    vals = [
        min(max(xc, 0.0), 1.0),
        min(max(yc, 0.0), 1.0),
        min(max(bw, 0.0), 1.0),
        min(max(bh, 0.0), 1.0),
    ]

    return str(cls_id) + " " + " ".join(f"{v:.6f}" for v in vals)


def polygon_to_yolo_line(cls_id, segmentation, img_w, img_h):
    # COCO polygon -> YOLO segmentation label.
    if not isinstance(segmentation, list):
        return None

    best_poly = None
    best_len = 0

    for poly in segmentation:
        if not isinstance(poly, list):
            continue

        if len(poly) < 6:
            continue

        if len(poly) > best_len:
            best_poly = poly
            best_len = len(poly)

    if best_poly is None:
        return None

    coords = []

    for i in range(0, len(best_poly), 2):
        x = best_poly[i]
        y = best_poly[i + 1]

        xn = min(max(float(x) / img_w, 0.0), 1.0)
        yn = min(max(float(y) / img_h, 0.0), 1.0)

        coords.extend([xn, yn])

    if len(coords) < 6:
        return None

    return str(cls_id) + " " + " ".join(f"{v:.6f}" for v in coords)


def write_yaml(out_dir, names, task_type):
    # YOLO dataset YAML.
    out_dir = Path(out_dir)

    yaml_data = {
        "path": str(out_dir),
        "train": "images/train",
        "val": "images/val",
        "nc": len(names),
        "names": names,
    }

    yaml_name = "coco_subset_det.yaml" if task_type == "det" else "coco_subset_seg.yaml"
    yaml_path = out_dir / yaml_name

    with open(yaml_path, "w") as f:
        yaml.safe_dump(yaml_data, f, sort_keys=False, allow_unicode=True)

    print("[YAML saved]", yaml_path)
    return yaml_path


def build_split(
    split_name,
    json_path,
    image_root,
    det_out_dir,
    seg_out_dir,
    max_images,
    copy_mode,
    seed,
):
    # Build one train/val split with filtered labels and linked/copied images.
    print(f"\n[{split_name}] loading json: {json_path}")

    with open(json_path, "r") as f:
        data = json.load(f)

    old_to_new, names = build_class_mapping(data["categories"])

    print(f"[{split_name}] number of classes: {len(names)}")
    print(f"[{split_name}] first 10 classes:")
    for i in range(min(10, len(names))):
        print(f"  {i}: {names[i]}")

    image_id_to_info = {}

    for img in data["images"]:
        image_path = Path(image_root) / img["file_name"]

        if image_path.exists():
            image_id_to_info[img["id"]] = {
                "file_name": img["file_name"],
                "path": image_path,
                "width": img["width"],
                "height": img["height"],
            }

    print(f"[{split_name}] existing images matched: {len(image_id_to_info)}")

    anns_by_image = defaultdict(list)

    for ann in tqdm(data["annotations"], desc=f"[{split_name}] scanning annotations"):
        image_id = ann.get("image_id")
        cat_id = ann.get("category_id")

        if image_id not in image_id_to_info:
            continue

        if cat_id not in old_to_new:
            continue

        if ann.get("iscrowd", 0) == 1:
            continue

        if "bbox" not in ann:
            continue

        if "segmentation" not in ann:
            continue

        if not isinstance(ann["segmentation"], list):
            continue

        anns_by_image[image_id].append(ann)

    candidate_ids = [img_id for img_id, anns in anns_by_image.items() if len(anns) > 0]

    rng = random.Random(seed)
    rng.shuffle(candidate_ids)

    if max_images > 0:
        candidate_ids = candidate_ids[:max_images]

    print(f"[{split_name}] selected images: {len(candidate_ids)}")

    det_img_out = Path(det_out_dir) / "images" / split_name
    det_label_out = Path(det_out_dir) / "labels" / split_name
    seg_img_out = Path(seg_out_dir) / "images" / split_name
    seg_label_out = Path(seg_out_dir) / "labels" / split_name

    for d in [det_img_out, det_label_out, seg_img_out, seg_label_out]:
        d.mkdir(parents=True, exist_ok=True)

    made_images = 0
    made_instances = 0

    for image_id in tqdm(candidate_ids, desc=f"[{split_name}] writing labels"):
        info = image_id_to_info[image_id]
        img_w = info["width"]
        img_h = info["height"]
        src_img = info["path"]
        file_name = info["file_name"]
        stem = Path(file_name).stem

        det_lines = []
        seg_lines = []

        for ann in anns_by_image[image_id]:
            cls_id = old_to_new[ann["category_id"]]

            det_line = coco_bbox_to_yolo_line(cls_id, ann["bbox"], img_w, img_h)
            seg_line = polygon_to_yolo_line(cls_id, ann["segmentation"], img_w, img_h)

            if det_line is None or seg_line is None:
                continue

            det_lines.append(det_line)
            seg_lines.append(seg_line)

        if not det_lines or not seg_lines:
            continue

        link_or_copy(src_img, det_img_out / file_name, copy_mode)
        link_or_copy(src_img, seg_img_out / file_name, copy_mode)

        with open(det_label_out / f"{stem}.txt", "w") as f:
            f.write("\n".join(det_lines) + "\n")

        with open(seg_label_out / f"{stem}.txt", "w") as f:
            f.write("\n".join(seg_lines) + "\n")

        made_images += 1
        made_instances += len(det_lines)

    print(f"[{split_name}] final made images: {made_images}")
    print(f"[{split_name}] final made instances: {made_instances}")

    return names, made_images, made_instances


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_json", required=True)
    parser.add_argument("--val_json", required=True)
    parser.add_argument("--train_img_root", required=True)
    parser.add_argument("--val_img_root", required=True)
    parser.add_argument("--det_out_dir", required=True)
    parser.add_argument("--seg_out_dir", required=True)
    parser.add_argument("--max_train", type=int, default=10000)
    parser.add_argument("--max_val", type=int, default=5000)
    parser.add_argument("--copy_mode", choices=["symlink", "copy"], default="symlink")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    det_out_dir = Path(args.det_out_dir)
    seg_out_dir = Path(args.seg_out_dir)

    det_out_dir.mkdir(parents=True, exist_ok=True)
    seg_out_dir.mkdir(parents=True, exist_ok=True)

    train_names, train_images, train_instances = build_split(
        split_name="train",
        json_path=args.train_json,
        image_root=args.train_img_root,
        det_out_dir=args.det_out_dir,
        seg_out_dir=args.seg_out_dir,
        max_images=args.max_train,
        copy_mode=args.copy_mode,
        seed=args.seed,
    )

    val_names, val_images, val_instances = build_split(
        split_name="val",
        json_path=args.val_json,
        image_root=args.val_img_root,
        det_out_dir=args.det_out_dir,
        seg_out_dir=args.seg_out_dir,
        max_images=args.max_val,
        copy_mode=args.copy_mode,
        seed=args.seed + 1,
    )

    if train_names != val_names:
        print("[WARN] train and val names are different.")

    write_yaml(det_out_dir, train_names, task_type="det")
    write_yaml(seg_out_dir, train_names, task_type="seg")

    print("\n[DONE]")
    print(f"Detection dataset: {det_out_dir}")
    print(f"Segmentation dataset: {seg_out_dir}")
    print(f"Train images: {train_images}, train instances: {train_instances}")
    print(f"Val images: {val_images}, val instances: {val_instances}")


if __name__ == "__main__":
    main()
