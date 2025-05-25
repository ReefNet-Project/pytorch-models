#!/usr/bin/env python3
"""
per_class_analysis.py

Usage
-----
python per_class_analysis.py --best_model "ViT-L/16-scratch"
"""
import argparse, json, glob, os, pandas as pd, matplotlib.pyplot as plt

###############################################################################
# cmd‑line args
###############################################################################
ap = argparse.ArgumentParser()
ap.add_argument("--best_model", default="vit_large_patch16_384_scratch",
                help="Model name (matches JSON filename prefix) to plot in detail")
args = ap.parse_args()

###############################################################################
# collect all *_summary.json files
###############################################################################
summary_files = glob.glob("evaluation_results/*_summary.json")
if not summary_files:
    raise FileNotFoundError("No *_summary.json files in the current directory.")

data = {}  # {class: {model: f1}}

for path in summary_files:
    model_name = os.path.basename(path).replace("_summary.json", "")
    with open(path) as f:
        js = json.load(f)
    for cls, metrics in js["per_class"].items():
        data.setdefault(cls, {})[model_name] = metrics["f1"]

df = pd.DataFrame(data).T.sort_index()   # rows = classes, cols = models
df.to_csv("per_class_f1.csv")
print("✅ per‑class F1 saved to per_class_f1.csv")

###############################################################################
# plot per‑class F1 for chosen model
###############################################################################
if args.best_model in df.columns:
    plt.figure(figsize=(10, 4))
    df[args.best_model].sort_values().plot(kind="bar")
    plt.ylabel("F1 score")
    plt.title(f"Per‑class F1 – {args.best_model}")
    plt.tight_layout()
    plt.savefig("f1_per_class_best_model.png", dpi=150)
    print("📈 per‑class plot saved → f1_per_class_best_model.png")
else:
    print(f"[warn] '{args.best_model}' not found in columns; available: {list(df.columns)}")

###############################################################################
# hardest 5 classes averaged across models
###############################################################################
hardest = df.mean(axis=1).sort_values().head(5)
plt.figure(figsize=(6, 3))
hardest.plot(kind="barh")
plt.xlabel("Mean F1 across models")
plt.title("Hardest coral genera (avg F1)")
plt.tight_layout()
plt.savefig("hardest_classes.png", dpi=150)
print("📈 hardest‑classes plot saved → hardest_classes.png")
