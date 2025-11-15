# utils/converter_para_yolo.py
import os
from pathlib import Path
from tqdm import tqdm
import cv2

# Caminhos base (ajuste se preciso)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_DIR = PROJECT_ROOT / "data" / "visdrone_raw"
OUT_BASE = PROJECT_ROOT / "data" / "visdrone"

SPLITS = {
    "train": {
        "images": BASE_DIR / "VisDrone2019-DET-train" / "images",
        "ann": BASE_DIR / "VisDrone2019-DET-train" / "annotations"
    },
    "val": {
        "images": BASE_DIR / "VisDrone2019-DET-val" / "images",
        "ann": BASE_DIR / "VisDrone2019-DET-val" / "annotations"
    },
    "test": {
        "images": BASE_DIR / "VisDrone2019-DET-test-dev" / "images",
        "ann": BASE_DIR / "VisDrone2019-DET-test-dev" / "annotations"
    }
}

# VisDrone classes (DET):
# 1: pedestrian, 2: people, 3: bicycle, 4: car, 5: van,
# 6: truck, 7: tricycle, 8: awning-tricycle, 9: bus,
# 10: motor, 11: others
CLASS_MAP = {
    1: 0,  # pedestrian
    2: 1,  # person
    3: 2,  # bicycle
    4: 3,  # car
    5: 4,  # van
    6: 5,  # truck
    9: 6,  # bus
}

def convert_bbox_to_yolo(x, y, w, h, img_w, img_h):
    x_center = x + w / 2.0
    y_center = y + h / 2.0

    return (
        x_center / img_w,
        y_center / img_h,
        w / img_w,
        h / img_h
    )

def process_split(split_name, cfg):
    images_dir = cfg["images"]
    ann_dir = cfg["ann"]

    out_img_dir = OUT_BASE / "images" / split_name
    out_lbl_dir = OUT_BASE / "labels" / split_name

    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    ann_files = list(ann_dir.glob("*.txt"))

    for ann_path in tqdm(ann_files, desc=f"Convertendo {split_name}"):
        img_name = ann_path.stem + ".jpg"
        img_path = images_dir / img_name

        if not img_path.exists():
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        out_img_path = out_img_dir / img_name
        if not out_img_path.exists():
            cv2.imwrite(str(out_img_path), img)

        yolo_lines = []
        with open(ann_path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 8:
                    continue

                x, y, w, h = map(float, parts[0:4])
                score = float(parts[4])
                cls_id = int(parts[5])

                if cls_id not in CLASS_MAP:
                    continue

                yolo_cls = CLASS_MAP[cls_id]
                x_c, y_c, w_n, h_n = convert_bbox_to_yolo(x, y, w, h, img_w, img_h)

                if w_n <= 0 or h_n <= 0:
                    continue

                yolo_lines.append(
                    f"{yolo_cls} {x_c:.6f} {y_c:.6f} {w_n:.6f} {h_n:.6f}"
                )

        out_lbl_path = out_lbl_dir / (ann_path.stem + ".txt")
        with open(out_lbl_path, "w") as f_out:
            f_out.write("\n".join(yolo_lines))

def main():
    for split, cfg in SPLITS.items():
        process_split(split, cfg)   

if __name__ == "__main__":
    main()
