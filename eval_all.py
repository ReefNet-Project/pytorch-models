#!/usr/bin/env python3
"""evaluate_coral_model.py

Evaluate a timm‑style image‑classification checkpoint on the ReefNet
coral‑patch dataset and export

* **per‑sample CSV** with top‑k predictions & confidences
* **JSON summary** with global and per‑class metrics

Designed for a **single GPU** (or CPU).  Handles checkpoints produced with
DDP by stripping the `module.` prefix.  To minimise GPU memory, logits are
immediately moved to the CPU; all softmax, top‑k, and metric computation runs
there.  Mixed‑precision (AMP) is **enabled by default** but can be disabled via
`--no-amp`.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm

from dataset_reefnet import CoralPatchCSVLoader  # your dataset class
from dataset_rsg import RSGPatchCSVLoader
################################################################################
# Data loader helper
################################################################################

def build_loader(csv_path: str, input_size: int, batch: int, workers: int) -> Tuple[DataLoader, CoralPatchCSVLoader]:
    """Return DataLoader (val/test split) and dataset instance."""
    transform = timm.data.create_transform(
        input_size=input_size,
        is_training=False,
        auto_augment=None,
    )

    # --- MAE-style *exact* test pipeline -----------------------------
    # transform = timm.data.create_transform(
    #     input_size=input_size,          # (3, 224|384, 224|384)
    #     is_training=False,
    #     crop_pct=1.0,                   # <-- **hard-coded**: no centre-crop
    #     interpolation="bicubic",        # MAE repo uses bicubic for 384-px ViT
    #     auto_augment=None,
    # )

    ds = CoralPatchCSVLoader(csv_path,is_train=False, is_test=True, transform=transform)
    ld = DataLoader(ds, batch_size=batch, num_workers=workers, pin_memory=True, shuffle=False)
    return ld, ds

################################################################################
# Validation loop (GPU‑light)
################################################################################

@torch.inference_mode()  # ➜ turns off autograd, huge memory savings
def validate(model: torch.nn.Module, loader: DataLoader, device: torch.device, use_amp: bool = True):
    """Run evaluation and collect tensors on CPU to keep GPU memory flat."""
    top1 = top5 = total = 0
    probs_list: List[torch.Tensor] = []
    targets_list: List[torch.Tensor] = []
    top5_preds_list: List[torch.Tensor] = []
    top5_confs_list: List[torch.Tensor] = []

    autocast_ctx = torch.amp.autocast(
        device_type="cuda", dtype=torch.float16,
        enabled=(use_amp and device.type == "cuda")
    )

    for x, y in tqdm(loader, unit="batch", desc="Validate"):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with autocast_ctx:
            logits = model(x)              # on GPU
        logits_cpu = logits.float().cpu()  # immediately to CPU
        del logits                         # release GPU memory

        probs_cpu = F.softmax(logits_cpu, dim=1)
        confs_cpu, preds_cpu = probs_cpu.topk(5, dim=1, largest=True, sorted=True)

        total += y.size(0)
        top1 += preds_cpu[:, 0].eq(y.cpu()).sum().item()
        top5 += preds_cpu.eq(y.cpu().view(-1, 1)).sum().item()

        probs_list.append(probs_cpu)
        targets_list.append(y.cpu())
        top5_preds_list.append(preds_cpu)
        top5_confs_list.append(confs_cpu)

    return (
        100 * top1 / total,
        100 * top5 / total,
        torch.cat(probs_list),
        torch.cat(targets_list),
        torch.cat(top5_preds_list),
        torch.cat(top5_confs_list),
    )

################################################################################
# Metric helpers
################################################################################

def compute_global_metrics(labels: np.ndarray, preds: np.ndarray):
    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
    micro_p, micro_r, micro_f1, _ = precision_recall_fscore_support(labels, preds, average="micro", zero_division=0)
    w_p, w_r, w_f1, _ = precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)
    return {
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "micro_precision": float(micro_p),
        "micro_recall": float(micro_r),
        "micro_f1": float(micro_f1),
        "weighted_precision": float(w_p),
        "weighted_recall": float(w_r),
        "weighted_f1": float(w_f1),
        "top1_accuracy": float((labels == preds).mean()),
    }

################################################################################
# Reporting
################################################################################

def evaluate_and_save(
    probs: torch.Tensor,
    targets: torch.Tensor,
    top5_preds: torch.Tensor,
    top5_confs: torch.Tensor,
    label_map: Dict[str, int],
    out_prefix: Path,
):
    """Write per‑sample CSV and global JSON summary."""
    preds = torch.argmax(probs, dim=1).cpu().numpy()
    labels = targets.cpu().numpy()
    probs_np = probs.cpu().numpy()
    top5_preds_np = top5_preds.cpu().numpy()
    top5_confs_np = top5_confs.cpu().numpy()

    # id→name list for fast lookup
    class_names = [None] * len(label_map)
    for name, idx in label_map.items():
        class_names[idx] = name

    top1_conf = probs_np[np.arange(len(preds)), preds]

    rows = []
    for i in range(len(labels)):
        row = {
            "ground_truth": class_names[labels[i]],
            "predicted_label": class_names[preds[i]],
            "top1_confidence": float(top1_conf[i]),
        }
        for k in range(5):
            row[f"top{k+1}_pred"] = class_names[top5_preds_np[i, k]]
            row[f"top{k+1}_conf"] = float(top5_confs_np[i, k])
        rows.append(row)

    csv_path = out_prefix.with_suffix(".csv")
    pd.DataFrame(rows).to_csv(csv_path, index=False, float_format="%.6g")
    print(f"✅ Per‑sample predictions → {csv_path}")

    summary = compute_global_metrics(labels, preds)
    all_classes = np.arange(len(class_names)) 
    per_p, per_r, per_f1, per_sup = precision_recall_fscore_support(labels, preds, labels=all_classes,average=None, zero_division=0)    
    # summary["per_class"] = {
    #     class_names[i]: {"precision": float(class_p[i]), "recall": float(class_r[i]), "f1": float(class_f1[i])}
    #     for i in range(len(class_names))
    # }
    summary["per_class"] = {}
    for i, cname in enumerate(class_names):
        if per_sup[i] == 0:            # <- skip absent class
            continue
        summary["per_class"][cname] = {
            "precision": float(per_p[i]),
            "recall":    float(per_r[i]),
            "f1":        float(per_f1[i]),
            "support":   int(per_sup[i]),
        }
    
    json_path = out_prefix.with_name(out_prefix.name + "_summary.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"✅ Summary metrics   → {json_path}")

################################################################################
# Main entry point
################################################################################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch", required=True, help="Model name for timm.create_model")
    parser.add_argument("--checkpoint", required=True, help="Path to *.pth.tar checkpoint")
    parser.add_argument("--csv", required=True, help="Dataset CSV containing val/test split")
    parser.add_argument("--input-size", type=int, default=224)
    parser.add_argument("-b", "--batch-size", type=int, default=64, help="Per‑GPU batch size")
    parser.add_argument("-j", "--workers", type=int, default=4)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed‑precision")
    parser.add_argument("--out", default="evaluation_report", help="Output prefix (no extension)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader, dataset = build_loader(args.csv, args.input_size, args.batch_size, args.workers)
    label_map = dataset.label_to_index

    model = timm.create_model(args.arch, num_classes=dataset.nb_classes, pretrained=False).to(device)
    # model = timm.create_model(
    #     args.arch,
    #     num_classes=dataset.nb_classes,
    #     pretrained=False,
    #     global_pool="avg",              # <-- **hard-coded**
    # ).to(device)

    # Load checkpoint (handles timm + DDP)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict") or ckpt.get("model") or ckpt
    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    model.eval()

    # Evaluate
    top1, top5, probs, labels, top5_preds, top5_confs = validate(model, loader, device, use_amp=not args.no_amp)
    print(f"{os.path.basename(args.checkpoint)} | top‑1: {top1:5.2f}%  top‑5: {top5:5.2f}%")

    evaluate_and_save(probs, labels, top5_preds, top5_confs, label_map, Path(args.out))


if __name__ == "__main__":
    main()
