"""
Generate dataset EDA figures + result tables for the Brain Tumor Detection project.

Reads MRI images from ../Data/brain_tumor_dataset/{yes,no} and writes:
  Figures/  16..21  — class balance, image-size distributions, mean images, intensity
  Tables/   01..04  — dataset composition, image-size stats, model comparison, per-model metrics
  Results/  model_metrics.csv / .json

Model metrics are derived from the confusion matrices produced in the notebook
(no retraining required). Uses numpy / matplotlib / Pillow only.
"""
import os, json
import numpy as np
from PIL import Image
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "Data", "brain_tumor_dataset")
FIG  = os.path.join(ROOT, "Figures"); os.makedirs(FIG, exist_ok=True)
TBL  = os.path.join(ROOT, "Tables");  os.makedirs(TBL, exist_ok=True)
RES  = os.path.join(ROOT, "Results"); os.makedirs(RES, exist_ok=True)

def md(df_rows, headers, path, caption):
    lines = [f"### {caption}", "", "| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in df_rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")
    open(path, "w").write("\n".join(lines) + "\n")

def csv(df_rows, headers, path):
    lines = [",".join(headers)] + [",".join(str(x) for x in r) for r in df_rows]
    open(path, "w").write("\n".join(lines) + "\n")

# ---------- scan images ----------
records = []  # (cls, w, h, mode, kb)
for cls in ("yes", "no"):
    d = os.path.join(DATA, cls)
    for f in os.listdir(d):
        p = os.path.join(d, f)
        try:
            with Image.open(p) as im:
                w, h = im.size; mode = im.mode
            kb = os.path.getsize(p) / 1024
            records.append((cls, w, h, mode, kb))
        except Exception:
            pass

cls_arr = np.array([r[0] for r in records])
W = np.array([r[1] for r in records]); H = np.array([r[2] for r in records])
KB = np.array([r[4] for r in records])
n_yes = int((cls_arr == "yes").sum()); n_no = int((cls_arr == "no").sum()); N = len(records)
print("images:", N, "| yes", n_yes, "| no", n_no)

# ---------- TABLE 1: dataset composition ----------
rows = [["Tumor (yes)", n_yes, f"{100*n_yes/N:.1f}"],
        ["Non-tumor (no)", n_no, f"{100*n_no/N:.1f}"],
        ["Total", N, "100.0"]]
md(rows, ["Class", "images", "pct"], os.path.join(TBL, "01_dataset_composition.md"), "Dataset Composition")
csv(rows, ["class", "images", "pct"], os.path.join(TBL, "01_dataset_composition.csv"))

# ---------- TABLE 2: image size stats ----------
rows = [["Width (px)",  int(W.min()), int(np.median(W)), int(W.max()), f"{W.mean():.0f}"],
        ["Height (px)", int(H.min()), int(np.median(H)), int(H.max()), f"{H.mean():.0f}"],
        ["File size (KB)", f"{KB.min():.1f}", f"{np.median(KB):.1f}", f"{KB.max():.1f}", f"{KB.mean():.1f}"]]
md(rows, ["metric", "min", "median", "max", "mean"], os.path.join(TBL, "02_image_size_stats.md"), "Image Size Statistics")
csv(rows, ["metric", "min", "median", "max", "mean"], os.path.join(TBL, "02_image_size_stats.csv"))

# ---------- FIG 16: class distribution ----------
fig, ax = plt.subplots(figsize=(6, 4))
ax.bar(["Tumor (yes)", "Non-tumor (no)"], [n_yes, n_no], color=["#c44e52", "#4c72b0"])
ax.set_title("Class Distribution"); ax.set_ylabel("Number of images")
for i, v in enumerate([n_yes, n_no]): ax.text(i, v + 1, str(v), ha="center")
fig.savefig(os.path.join(FIG, "16_class_distribution.png"), bbox_inches="tight"); plt.close(fig)

# ---------- FIG 17: width vs height scatter ----------
fig, ax = plt.subplots(figsize=(7, 6))
for cls, col in [("yes", "#c44e52"), ("no", "#4c72b0")]:
    m = cls_arr == cls
    ax.scatter(W[m], H[m], s=14, alpha=0.5, c=col, label=cls)
ax.set_title("Image Dimensions (width vs height)"); ax.set_xlabel("Width (px)"); ax.set_ylabel("Height (px)"); ax.legend()
fig.savefig(os.path.join(FIG, "17_image_dimensions_scatter.png"), bbox_inches="tight"); plt.close(fig)

# ---------- FIG 18: width/height histograms ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].hist(W, bins=30, color="#4c72b0"); axes[0].set_title("Width distribution"); axes[0].set_xlabel("px")
axes[1].hist(H, bins=30, color="#55a868"); axes[1].set_title("Height distribution"); axes[1].set_xlabel("px")
fig.savefig(os.path.join(FIG, "18_image_size_histograms.png"), bbox_inches="tight"); plt.close(fig)

# ---------- FIG 19: aspect ratio ----------
ar = W / H
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(ar, bins=30, color="#8172b3"); ax.axvline(1.0, color="k", ls="--", lw=1)
ax.set_title("Aspect Ratio (width / height)"); ax.set_xlabel("ratio")
fig.savefig(os.path.join(FIG, "19_aspect_ratio.png"), bbox_inches="tight"); plt.close(fig)

# ---------- FIG 20: mean image per class ----------
def mean_image(cls, size=128):
    d = os.path.join(DATA, cls); acc = np.zeros((size, size)); n = 0
    for f in os.listdir(d):
        try:
            with Image.open(os.path.join(d, f)) as im:
                acc += np.asarray(im.convert("L").resize((size, size)), float); n += 1
        except Exception: pass
    return acc / max(n, 1)
fig, axes = plt.subplots(1, 2, figsize=(9, 5))
axes[0].imshow(mean_image("yes"), cmap="gray"); axes[0].set_title("Mean image — Tumor"); axes[0].axis("off")
axes[1].imshow(mean_image("no"),  cmap="gray"); axes[1].set_title("Mean image — Non-tumor"); axes[1].axis("off")
fig.suptitle("Average MRI per Class")
fig.savefig(os.path.join(FIG, "20_mean_image_per_class.png"), bbox_inches="tight"); plt.close(fig)

# ---------- FIG 21: pixel intensity by class ----------
def sample_pixels(cls, size=64, cap=80):
    d = os.path.join(DATA, cls); vals = []
    for f in os.listdir(d)[:cap]:
        try:
            with Image.open(os.path.join(d, f)) as im:
                vals.append(np.asarray(im.convert("L").resize((size, size)), float).ravel())
        except Exception: pass
    return np.concatenate(vals) if vals else np.array([])
fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(sample_pixels("yes"), bins=50, alpha=0.5, color="#c44e52", label="Tumor", density=True)
ax.hist(sample_pixels("no"),  bins=50, alpha=0.5, color="#4c72b0", label="Non-tumor", density=True)
ax.set_title("Pixel Intensity Distribution by Class"); ax.set_xlabel("Grayscale value"); ax.legend()
fig.savefig(os.path.join(FIG, "21_pixel_intensity_by_class.png"), bbox_inches="tight"); plt.close(fig)

# ---------- Model metrics from confusion matrices (from notebook) ----------
# CM layout: [[TN, FP], [FN, TP]] with positive = tumor (class 1)
cms = {
    "CNN - 3 conv + early stopping": [[23, 2], [0, 35]],
    "CNN - 2 conv blocks":           [[23, 0], [0, 37]],
    "CNN - 3 conv + augmentation":   [[24, 0], [0, 36]],
    "VGG16 (transfer learning)":     [[4, 15], [8, 23]],
}
auc = {"CNN - 3 conv + early stopping": 0.98, "CNN - 2 conv blocks": 1.00,
       "CNN - 3 conv + augmentation": 1.00, "VGG16 (transfer learning)": 0.54}

def metrics(cm):
    (tn, fp), (fn, tp) = cm
    n = tn + fp + fn + tp
    acc = (tp + tn) / n
    prec = tp / (tp + fp) if tp + fp else 0
    rec = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0
    return acc, prec, rec, f1

rows_cmp, rows_full, results = [], [], {}
for name, cm in cms.items():
    acc, prec, rec, f1 = metrics(cm)
    rows_cmp.append([name, f"{acc:.3f}", f"{auc[name]:.2f}"])
    rows_full.append([name, f"{acc:.3f}", f"{prec:.3f}", f"{rec:.3f}", f"{f1:.3f}", f"{auc[name]:.2f}"])
    results[name] = {"accuracy": round(acc, 3), "precision": round(prec, 3),
                     "recall": round(rec, 3), "f1": round(f1, 3), "roc_auc": auc[name],
                     "confusion_matrix": cm}

md(rows_cmp, ["Model", "Accuracy", "ROC AUC"], os.path.join(TBL, "03_model_comparison.md"), "Model Comparison")
csv(rows_cmp, ["model", "accuracy", "roc_auc"], os.path.join(TBL, "03_model_comparison.csv"))
md(rows_full, ["Model", "Accuracy", "Precision", "Recall", "F1", "ROC AUC"],
   os.path.join(TBL, "04_per_model_metrics.md"), "Per-Model Metrics (tumor = positive class)")
csv(rows_full, ["model", "accuracy", "precision", "recall", "f1", "roc_auc"],
    os.path.join(TBL, "04_per_model_metrics.csv"))
json.dump(results, open(os.path.join(RES, "model_metrics.json"), "w"), indent=2)
csv(rows_full, ["model", "accuracy", "precision", "recall", "f1", "roc_auc"],
    os.path.join(RES, "model_metrics.csv"))
print("done — figures, tables, results written")
