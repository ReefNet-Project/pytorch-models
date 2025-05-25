#!/usr/bin/env python3
import argparse, os, json, torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import timm
from dataset_reefnet import CoralPatchCSVLoader
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, top_k_accuracy_score

def build_loader(csv, input_size, batch, workers):
    transform = timm.data.create_transform(
        input_size=input_size,
        is_training=False,
        auto_augment=None,
    )
    ds  = CoralPatchCSVLoader(csv, is_train=False, is_test=True, transform=transform)
    ld  = DataLoader(ds, batch_size=batch, num_workers=workers,
                     pin_memory=True, shuffle=False)
    return ld, ds

@torch.no_grad()
def validate(model, loader, device):
    top1 = top5 = total = 0
    all_probs, all_targets = [], []
    all_top5_preds, all_top5_confs = [], []

    for x, y in tqdm(loader, total=len(loader), unit='batch'):
        x, y = x.to(device, non_blocking=True), y.to(device)
        logits = model(x)
        probs = F.softmax(logits, dim=1)

        confs, preds = logits.topk(5, 1, True, True)  # (B, 5)
        total += y.size(0)
        top1 += preds[:, 0].eq(y).sum().item()
        top5 += preds.eq(y.view(-1, 1)).sum().item()

        all_probs.append(probs.cpu())
        all_targets.append(y.cpu())
        all_top5_preds.append(preds.cpu())
        all_top5_confs.append(F.softmax(confs, dim=1).cpu())

    return (
        100 * top1 / total,
        100 * top5 / total,
        torch.cat(all_probs),
        torch.cat(all_targets),
        torch.cat(all_top5_preds),
        torch.cat(all_top5_confs),
    )

def evaluate_and_save(probs, labels, top5_preds, top5_confs, label_map, output_path):
    preds = torch.argmax(probs, dim=1).numpy()
    labels = labels.numpy()
    probs = probs.numpy()
    top5_preds = top5_preds.numpy()
    top5_confs = top5_confs.numpy()

    class_names = [None] * len(label_map)
    for k, v in label_map.items():
        class_names[v] = k

    top1_conf = probs[np.arange(len(preds)), preds]

    macro_metrics = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    micro_metrics = precision_recall_fscore_support(labels, preds, average='micro', zero_division=0)
    weighted_metrics = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=0)

    per_class_p, per_class_r, per_class_f1, _ = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )

    # Build per-sample rows
    rows = []
    for i in range(len(labels)):
        gt = labels[i]
        pred = preds[i]
        row = {
            'ground_truth': class_names[gt],
            'predicted_label': class_names[pred],
            'top1_confidence': top1_conf[i]
        }
        for k in range(5):
            row[f'top{k+1}_pred'] = class_names[top5_preds[i, k]]
            row[f'top{k+1}_conf'] = top5_confs[i, k]

        # Append global metrics
        row.update({
            'macro_precision': macro_metrics[0],
            'macro_recall': macro_metrics[1],
            'macro_f1': macro_metrics[2],
            'micro_precision': micro_metrics[0],
            'micro_recall': micro_metrics[1],
            'micro_f1': micro_metrics[2],
            'weighted_precision': weighted_metrics[0],
            'weighted_recall': weighted_metrics[1],
            'weighted_f1': weighted_metrics[2],
        })

        # Per-class metrics for GT label
        row['class_precision'] = per_class_p[gt]
        row['class_recall'] = per_class_r[gt]
        row['class_f1'] = per_class_f1[gt]

        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"✅ Evaluation report saved to {output_path}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--arch',        required=True)
    ap.add_argument('--checkpoint',  required=True)
    ap.add_argument('--csv',         required=True)
    ap.add_argument('--input-size',  type=int, default=224)
    ap.add_argument('-b', '--batch-size', type=int, default=64)
    ap.add_argument('-j', '--workers',    type=int, default=3)
    ap.add_argument('--output-csv', default='evaluation_report.csv')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    loader, dataset = build_loader(args.csv, args.input_size,
                                   args.batch_size, args.workers)
    label_map = dataset.label_to_index
    model = timm.create_model(args.arch,
                              num_classes=dataset.nb_classes,
                              pretrained=False).to(device)

    ckpt = torch.load(
        args.checkpoint,
        map_location='cpu',
        weights_only=False
    )
    state = (ckpt.get("state_dict") or ckpt.get("model") or ckpt)

    if any(k.startswith("module.") for k in state):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state, strict=False)

    model.eval()
    top1, top5, all_probs, all_labels, all_top5_preds, all_top5_confs = validate(model, loader, device)

    print(f'{os.path.basename(args.checkpoint)}  '
          f'| top‑1: {top1:5.2f}%  top‑5: {top5:5.2f}%')

    evaluate_and_save(
        all_probs,
        all_labels,
        all_top5_preds,
        all_top5_confs,
        label_map,
        args.output_csv
    )

if __name__ == '__main__':
    main()
