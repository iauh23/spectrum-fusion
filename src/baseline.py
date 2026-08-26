import os
import csv
import json
from ultralytics import YOLO
from datetime import datetime
from tqdm import tqdm
from detect import get_detections
from matcher import match_frame_detections
from metrics import compute_frame_metric, compute_metrics


def run_baseline_pipeline(modality, model_path, image_dir, gt_dict):
    model = YOLO(model_path)
    all_frame_results = []
    failed_frames = []
    total_detections = 0
    total_gt_points = 0

    for image_id, gt_points in tqdm(gt_dict.items(), desc=f"{modality} baseline", unit="frame"):
        # Modality-aware filename: thermal keeps the "R" suffix (1R.jpg), RGB strips it (1.jpg)
        if modality == "thermal":
            image_name = f"{image_id}.jpg"
        else:
            image_name = f"{image_id.rstrip('R')}.jpg"

        image_path = os.path.join(image_dir, image_name)

        if not os.path.exists(image_path):
            failed_frames.append(f"Missing: {image_path}")
            continue

        detections = get_detections(model, image_path, is_thermal=(modality == "thermal"))
        match_result = match_frame_detections(gt_points, detections, modality=modality)

        total_detections += len(detections)
        total_gt_points += len(gt_points)

        frame_metric = compute_frame_metric(
            tp=match_result['tp'],
            fp=match_result['fp'],
            fn=match_result['fn'],
            match_records=match_result['match_records'],
        )

        frame_metric['image_id'] = image_id
        frame_metric['modality'] = modality
        all_frame_results.append(frame_metric)

    aggregated = compute_metrics(all_frame_results)

    # Recall diagnostics: how many people does the model even propose vs how many exist
    frames_processed = len(all_frame_results)
    diagnostics = {
        "frames_processed": frames_processed,
        "frames_failed": len(failed_frames),
        "total_gt_people": total_gt_points,
        "total_detections": total_detections,
        "avg_gt_per_frame": round(total_gt_points / frames_processed, 1) if frames_processed else 0,
        "avg_detections_per_frame": round(total_detections / frames_processed, 1) if frames_processed else 0,
    }
    print(f"\n[{modality}] diagnostics: {json.dumps(diagnostics)}")

    # Create the output directory up front so the failed-frames log can live in it
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(r'C:\Users\ia443\Desktop\Project\outputs\baselines', f'{modality}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)

    # Per-frame CSV (guarded against an empty run; 'distances' is a list, so ignore it)
    csv_path = os.path.join(output_dir, "per_frame_metrics.csv")
    if all_frame_results:
        fieldnames = [k for k in all_frame_results[0].keys() if k != 'distances']
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            writer.writerows(all_frame_results)
    else:
        with open(csv_path, "w", newline="") as f:
            f.write("")

    # Failed frames log (now written inside the output dir)
    if failed_frames:
        with open(os.path.join(output_dir, "failed_frames.log"), "w") as f:
            f.write("\n".join(failed_frames) + "\n")

    summary = {
        "metrics": aggregated,
        "diagnostics": diagnostics,
        "config": {
            "model_path": model_path,
            "conf_threshold": 0.30,
            "iou_threshold": 0.45,
            "img_size": 1280,
            "modality": modality,
            "timestamp": timestamp,
        },
    }

    json_path = os.path.join(output_dir, "summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return aggregated


if __name__ == "__main__":
    from gt_parser import gt_box

    gt_dict = gt_box()

    # Run 1: RGB
    print("Running RGB baseline...")
    rgb_result = run_baseline_pipeline(
        modality="rgb",
        model_path=r"C:\Users\ia443\Desktop\Project\models\yolov8x.pt",
        image_dir=r"C:\Users\ia443\Desktop\Project\data\rgb",
        gt_dict=gt_dict,
    )
    print("RGB complete:", rgb_result)

    # Run 2: Thermal (thermal-fine-tuned weights — an RGB COCO model on infrared is a placeholder, not a baseline)
    print("Running Thermal baseline...")
    thermal_result = run_baseline_pipeline(
        modality="thermal",
        model_path=r"C:\Users\ia443\Desktop\Project\models\thermal_yolov8n_human.pt",
        image_dir=r"C:\Users\ia443\Desktop\Project\data\infrared",
        gt_dict=gt_dict,
    )
    print("Thermal complete:", thermal_result)

    # Build REFERENCE.md
    baselines_dir = r"C:\Users\ia443\Desktop\Project\outputs\baselines"

    def find_latest(prefix):
        folders = [
            d for d in os.listdir(baselines_dir)
            if os.path.isdir(os.path.join(baselines_dir, d)) and d.startswith(prefix)
        ]
        if not folders:
            return None
        folders.sort()
        return os.path.join(baselines_dir, folders[-1])

    def fmt(v):
        if v is None:
            return "N/A"
        return f"{v:.4f}"

    ref_path = os.path.join(baselines_dir, "REFERENCE.md")
    with open(ref_path, "w") as f:
        f.write("# Baseline Reference Table\n\n")
        f.write("Pinned config for all runs: conf=0.30, iou=0.45, imgsz=1280, classes=person.\n\n")
        f.write("Modality | Model | Precision | Recall | F1 | MAE (px) | RMSE (px) | TP | FP | FN\n")
        f.write("--- | --- | --- | --- | --- | --- | --- | --- | --- | ---\n")

        notes = []
        for prefix, label in [("rgb_", "RGB-only"), ("thermal_", "Thermal-only")]:
            folder = find_latest(prefix)
            if folder is None:
                f.write(f"{label} | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A | N/A\n")
                continue
            with open(os.path.join(folder, "summary.json")) as fh:
                summary = json.load(fh)
            m = summary["metrics"]
            model_name = os.path.basename(summary["config"]["model_path"])
            f.write(
                f"{label} | {model_name} | {fmt(m['precision'])} | {fmt(m['recall'])} | "
                f"{fmt(m['f1'])} | {fmt(m['mae'])} | {fmt(m['rmse'])} | "
                f"{m['total_tp']} | {m['total_fp']} | {m['total_fn']}\n"
            )
            diag = summary.get("diagnostics")
            if diag:
                notes.append(
                    f"- **{label}**: {diag['frames_processed']} frames, "
                    f"avg {diag['avg_gt_per_frame']} GT people/frame vs "
                    f"{diag['avg_detections_per_frame']} detections/frame"
                )

        f.write("\n## Diagnostics\n\n")
        f.write("\n".join(notes) + "\n")
        f.write(
            "\n> Note: the thermal baseline uses thermal-fine-tuned weights "
            "(yolov8n capacity); the RGB baseline uses yolov8x. Capacity differs by design — "
            "domain match matters more than parameter count for a modality baseline.\n"
        )

    print(f"REFERENCE.md saved to {ref_path}")
