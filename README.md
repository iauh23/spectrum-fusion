# spectrum-fusion

**Person detection across RGB and thermal infrared, with point-level ground-truth matching, frame-weighted metrics, and fusion.**

An evaluation harness that compares two person-detection baselines — a visible-light (RGB) model and a thermal infrared model — against **point-level** ground truth, and lays the groundwork for fusing the two modalities.

---

## What this project does

Most detection benchmarks use bounding-box ground truth. This one uses **points**: each person is annotated as a single `(x, y)` location. The harness runs YOLOv8 over every image, matches each detection box to a ground-truth point, and reports both *classification* metrics (did we find the person?) and *localization* metrics (how close was the box to the person?).

The pipeline, end to end:

1. **Parse** XML point annotations → `src/gt_parser.py`
2. **Detect** people with YOLOv8 → `src/detect.py`
3. **Match** boxes to points (greedy, confidence-sorted, closest to box center) → `src/matcher.py`
4. **Score** precision / recall / F1 + MAE / RMSE → `src/metrics.py`
5. **Run** both baselines and save CSV + JSON + a comparison table → `src/baseline.py`

---

## Why I built this

Thermal infrared can see people in darkness, smoke, and poor lighting where a normal RGB camera fails — but it's not obvious whether thermal or RGB is better for detecting people in crowded scenes. I wanted a **fair, reproducible way to compare them**, and a foundation I could later extend to **fuse** the two for better results than either alone.

## The problem it solves

- **Point vs. box mismatch.** Crowd-counting datasets annotate people as points, not boxes, so standard box-IoU metrics don't apply. This harness handles point ground truth directly.
- **Apples-to-apples comparison.** Same thresholds, same image size, same matching logic for both modalities — so the RGB and thermal numbers mean the same thing.
- **Metric ambiguity.** Person-level (micro) vs. frame-level (macro) averaging give different answers. This project makes the choice explicit and documented.

## What makes it stand out

- **Point-in-box matching** — detections are matched to GT *points*, sorted by confidence and snapped to the closest point, one-to-one.
- **Frame-weighted (macro) metrics** — every scene counts equally, because getting the *scene* right matters more than counting *people* right in a crowd.
- **Localization, not just detection** — MAE / RMSE in pixels tell you how *precisely* each box lands on the person.
- **Modality-aware file pairing** — thermal keeps the `R` suffix (`1R.jpg`), RGB strips it (`1.jpg`), handled automatically.
- **Built-in recall diagnostics** — every run reports GT people/frame vs. detections/frame, so you can tell *why* recall is low (few proposals vs. bad placement).
- **Clean output** — per-frame CSV, a `summary.json`, and a `REFERENCE.md` comparison table, plus `tqdm` progress bars.

---

## Results

100 frames per modality, 6,272 total ground-truth people (~62.7 per frame).

| Modality | Model | Precision | Recall | F1 | MAE (px) | RMSE (px) | TP | FN |
|---|---|---|---|---|---|---|---|---|
| RGB | `yolov8x.pt` | 0.769 | 0.192 | 0.355 | 5.9 | 6.5 | 1216 | 5056 |
| Thermal | `thermal_yolov8n_human.pt` | 0.851 | 0.313 | 0.427 | 6.0 | 6.9 | 1709 | 4563 |

**The headline finding:** the thermal model — a *nano* architecture with ~3M parameters — beats the RGB `yolov8x` (~68M parameters) on F1. A domain-matched model wins over a generic one, even at ~20× smaller. This is the "right training data > model size" lesson in practice.

**The open problem:** both models find only ~15–20 people per frame out of ~63 present. Precision and localization are strong; the bottleneck is that the models simply don't propose enough boxes in dense crowds.

---

## What I learned

- **Domain-matched training beats model size.** Swapping a generic RGB model for a thermal-fine-tuned model tripled thermal recall (0.11 → 0.31) and flipped the ranking.
- **Ground-truth format changes the whole matching story.** Points need point-in-box matching, not IoU.
- **Metrics are a design decision.** Frame-weighted vs. person-weighted averaging changes the numbers — you have to choose deliberately and document it.
- **Diagnostics before tuning.** Logging GT/frame vs. detections/frame immediately revealed *why* recall was low, instead of guessing.

## Tech stack — why these tools

- **Python 3.11** — the environment where `ultralytics` is installed.
- **Ultralytics YOLOv8** — one consistent API for both the generic RGB weights and the thermal-fine-tuned weights.
- **OpenCV** — image loading (grayscale for thermal, BGR for RGB).
- **NumPy** — thermal channel handling.
- **tqdm** — progress bars so long runs aren't a black box.

## Challenges I faced

- **Thermal model mismatch** — pointing an RGB-trained model at infrared produced garbage (recall 0.11). Fixed by using a domain-matched thermal model.
- **Encoding bugs** — stray non-UTF-8 characters (em-dashes) broke compilation.
- **CSV field mismatch** — a per-frame `distances` list didn't fit the CSV schema; fixed with `extrasaction='ignore'`.
- **Dense crowds** — ~63 people/frame with heavy occlusion means low recall; this is the current frontier.

## Features I hope to implement

- **RGB + thermal fusion** — combine detections from both modalities (the whole reason this harness exists).
- **Overlap analysis** — measure whether RGB and thermal miss the *same* people or *different* ones, to know how much fusion can actually gain.
- **Confidence-threshold sweep** — find whether lowering `conf` recovers the missed crowd without wrecking precision.

---

## How to install and run

Requires **Python 3.11**.

```bash
git clone https://github.com/iauh23/spectrum-fusion.git
cd spectrum-fusion
pip install -r requirements.txt
```

Then download the model weights (see **Model downloads** below) into `models/`, and run:

```bash
python src/baseline.py
```

Output lands in `outputs/baselines/` — a timestamped folder per modality with `per_frame_metrics.csv` and `summary.json`, plus a `REFERENCE.md` table.

## How to use

Data layout:

```
data/
  rgb/        # 100 visible-light images:  1.jpg ... 100.jpg
  infrared/   # 100 thermal images:        1R.jpg ... 100R.jpg
  GT/         # 100 point annotations:     1R.xml ... 100R.xml
```

Naming convention: a GT id of `NR` maps to thermal image `NR.jpg` and RGB image `N.jpg` (the trailing `R` is stripped for RGB). `baseline.py` handles this automatically.

To change models or thresholds, edit the paths in `src/baseline.py` and the `conf` / `iou` / `imgsz` values in `src/detect.py`.

## Project structure

```
src/
  gt_parser.py   # parse XML point ground truth
  detect.py      # YOLOv8 detection (conf/iou/imgsz live here)
  matcher.py     # point-in-box greedy matching
  metrics.py     # frame-weighted P/R/F1 + MAE/RMSE
  baseline.py    # run both baselines, write outputs
data/            # rgb, infrared, GT
models/          # (gitignored) YOLO weights
outputs/         # (gitignored) results
```

## Model downloads

Weights are **not** committed (they're large binaries), so fetch them into `models/`:

- `yolov8x.pt` — generic RGB person detector, from [Ultralytics](https://github.com/ultralytics/ultralytics) (auto-downloads via `ultralytics` on first use).
- `thermal_yolov8n_human.pt` — thermal-fine-tuned, class 0 = human, from `pitangent-ds/YOLOv8-human-detection-thermal`.
