import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader

TRAINING_DIR = Path(__file__).resolve().parents[1] / "training"
sys.path.insert(0, str(TRAINING_DIR))
from dataset import ThymomaDataset
from model import ThymomaTransformerClassifier


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PACKAGE_ROOT / "data" / "external" / "images"
JSON_ROOT = PACKAGE_ROOT / "data" / "external" / "json"
DEFAULT_EVAL_SETS = [
    "external_shuguang.json",
    "external_huangshan.json",
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--result-dir", type=str, required=True)
    parser.add_argument("--data-root", type=str, default=DATA_ROOT)
    parser.add_argument("--json-root", type=str, default=JSON_ROOT)
    parser.add_argument("--eval-jsons", type=str, nargs="+", default=DEFAULT_EVAL_SETS)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--max-slices", type=int, default=32)
    parser.add_argument("--drop-out", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def infer_num_layers(state_dict):
    layer_ids = set()
    for key in state_dict:
        prefix = "transformer.layers."
        if key.startswith(prefix):
            rest = key[len(prefix):]
            layer_id = rest.split(".", 1)[0]
            if layer_id.isdigit():
                layer_ids.add(int(layer_id))
    if not layer_ids:
        raise ValueError("Could not infer num_layers from checkpoint.")
    return max(layer_ids) + 1


def load_model(model_path, device, max_slices, drop_out):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    num_layers = infer_num_layers(state_dict)
    model = ThymomaTransformerClassifier(
        backbone_name="vit_tiny_patch16_224",
        num_classes=2,
        max_slices=max_slices,
        drop_out=drop_out,
        num_layers=num_layers,
        backbone_checkpoint=None,
    )
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model, checkpoint, num_layers


@torch.no_grad()
def predict_dataset(model, data_root, json_path, image_size, max_slices, batch_size, num_workers, device):
    dataset = ThymomaDataset(
        image_root=data_root,
        label_json=json_path,
        is_train=False,
        max_slices=max_slices,
        target_slices=max_slices,
        image_size=image_size,
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    all_probs = []
    all_preds = []
    all_labels = []
    all_pids = []

    pid_order = dataset.patient_ids
    seen = 0
    total_batches = len(dataloader)
    for batch_idx, (imgs, labels) in enumerate(dataloader, start=1):
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        labels_np = labels.numpy()
        batch_size_now = len(labels_np)
        batch_pids = pid_order[seen:seen + batch_size_now]
        seen += batch_size_now

        all_probs.append(probs)
        all_preds.append(preds)
        all_labels.append(labels_np)
        all_pids.extend(batch_pids)
        if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == total_batches:
            print(f"  batch {batch_idx}/{total_batches}")

    probs = np.concatenate(all_probs, axis=0)
    preds = np.concatenate(all_preds, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return {
        "patient_ids": all_pids,
        "y_true": labels,
        "y_pred": preds,
        "probs": probs,
    }


def compute_metrics(y_true, y_pred, probs):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, probs[:, 1])
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    return {
        "n": int(len(y_true)),
        "ACC": float(acc),
        "AUC": float(auc),
        "Sens": float(sens),
        "Spec": float(spec),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def save_predictions(path, result):
    rows = []
    probs = result["probs"]
    for pid, y_true, y_pred, prob in zip(result["patient_ids"], result["y_true"], result["y_pred"], probs[:, 1]):
        rows.append(
            {
                "patient_id": pid,
                "y_true": int(y_true),
                "y_pred": int(y_pred),
                "prob_class1": float(prob),
            }
        )
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    os.makedirs(args.result_dir, exist_ok=True)
    device = torch.device(args.device)

    model, checkpoint, num_layers = load_model(
        model_path=args.model_path,
        device=device,
        max_slices=args.max_slices,
        drop_out=args.drop_out,
    )

    summary_rows = []
    combined_y_true = []
    combined_y_pred = []
    combined_probs = []

    for eval_json in args.eval_jsons:
        json_path = eval_json if os.path.isabs(eval_json) else os.path.join(args.json_root, eval_json)
        dataset_name = os.path.splitext(os.path.basename(json_path))[0]
        print(f"Running {dataset_name} ...")
        result = predict_dataset(
            model=model,
            data_root=args.data_root,
            json_path=json_path,
            image_size=args.image_size,
            max_slices=args.max_slices,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
        )
        metrics = compute_metrics(result["y_true"], result["y_pred"], result["probs"])
        metrics["dataset"] = dataset_name
        summary_rows.append(metrics)

        combined_y_true.append(result["y_true"])
        combined_y_pred.append(result["y_pred"])
        combined_probs.append(result["probs"])

        save_predictions(os.path.join(args.result_dir, f"{dataset_name}_predictions.json"), result)
        print(
            f"{dataset_name}: n={metrics['n']}, ACC={metrics['ACC']:.4f}, "
            f"AUC={metrics['AUC']:.4f}, Sens={metrics['Sens']:.4f}, Spec={metrics['Spec']:.4f}"
        )

    combined_metrics = compute_metrics(
        np.concatenate(combined_y_true, axis=0),
        np.concatenate(combined_y_pred, axis=0),
        np.concatenate(combined_probs, axis=0),
    )
    combined_metrics["dataset"] = "external_combined"
    summary_rows.append(combined_metrics)

    summary_path = os.path.join(args.result_dir, "summary.csv")
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["dataset", "n", "ACC", "AUC", "Sens", "Spec", "TN", "FP", "FN", "TP"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    meta = {
        "model_path": args.model_path,
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_metrics": {
            key: checkpoint[key]
            for key in ("val_acc", "val_auc")
            if key in checkpoint
        },
        "num_layers": num_layers,
        "max_slices": args.max_slices,
        "drop_out": args.drop_out,
        "eval_jsons": args.eval_jsons,
    }
    with open(os.path.join(args.result_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(
        f"external_combined: n={combined_metrics['n']}, ACC={combined_metrics['ACC']:.4f}, "
        f"AUC={combined_metrics['AUC']:.4f}, Sens={combined_metrics['Sens']:.4f}, Spec={combined_metrics['Spec']:.4f}"
    )
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    main()
