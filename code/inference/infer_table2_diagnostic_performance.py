import argparse
import csv
import html
import importlib.util
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RUN_DIR = PACKAGE_ROOT / "models"
DEFAULT_CODE_DIR = PACKAGE_ROOT / "code" / "training"
DEFAULT_EXTERNAL_JSON_ROOT = PACKAGE_ROOT / "data" / "external" / "json"
DEFAULT_EXTERNAL_IMAGE_ROOT = PACKAGE_ROOT / "data" / "external" / "images"
DEFAULT_EXTERNAL_EVAL_SETS = ["external_shuguang.json", "external_huangshan.json"]
DEFAULT_PROSPECTIVE_JSON = PACKAGE_ROOT / "data" / "prospective" / "json" / "forward_ct.json"
DEFAULT_PROSPECTIVE_IMAGE_ROOT = PACKAGE_ROOT / "data" / "prospective" / "images"
DEFAULT_OUTPUT_DIR = PACKAGE_ROOT / "metrics"


def display_path(path):
    path = Path(path).resolve()
    try:
        return str(path.relative_to(PACKAGE_ROOT))
    except ValueError:
        return str(path)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run checkpoint inference on train/test, external centers, "
            "and prospective data, then generate a Table 2 style HTML/CSV/JSON."
        )
    )
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--model-name", default="best.pth")
    parser.add_argument("--code-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--output-prefix", default="table2_run6_diagnostic_performance")
    parser.add_argument("--external-json-root", type=Path, default=DEFAULT_EXTERNAL_JSON_ROOT)
    parser.add_argument("--external-image-root", type=Path, default=DEFAULT_EXTERNAL_IMAGE_ROOT)
    parser.add_argument("--external-eval-sets", nargs="+", default=DEFAULT_EXTERNAL_EVAL_SETS)
    parser.add_argument("--prospective-json", type=Path, default=DEFAULT_PROSPECTIVE_JSON)
    parser.add_argument("--prospective-image-root", type=Path, default=DEFAULT_PROSPECTIVE_IMAGE_ROOT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--drop-out", type=float, default=0.1)
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def load_module(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def infer_num_layers(state_dict):
    layer_ids = set()
    prefix = "transformer.layers."
    for key in state_dict:
        if key.startswith(prefix):
            layer_id = key[len(prefix):].split(".", 1)[0]
            if layer_id.isdigit():
                layer_ids.add(int(layer_id))
    if not layer_ids:
        raise ValueError("Could not infer transformer num_layers from checkpoint.")
    return max(layer_ids) + 1


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model(model_cls, model_path, device, max_slices, drop_out):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    num_layers = infer_num_layers(state_dict)
    model = model_cls(
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
def predict_dataset(
    model,
    dataset_cls,
    cohort_name,
    image_root,
    json_path,
    image_size,
    max_slices,
    batch_size,
    num_workers,
    device,
    threshold,
):
    dataset = dataset_cls(
        image_root=str(image_root),
        label_json=str(json_path),
        is_train=False,
        max_slices=max_slices,
        target_slices=max_slices,
        image_size=image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=str(device).startswith("cuda"),
    )

    patient_ids = dataset.patient_ids
    seen = 0
    rows = []
    y_true_parts = []
    y_pred_parts = []
    y_score_parts = []

    for batch_idx, (imgs, labels) in enumerate(loader, start=1):
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()
        scores = probs[:, 1]
        preds = (scores >= threshold).astype(np.int64)
        labels_np = labels.numpy().astype(np.int64)
        batch_pids = patient_ids[seen:seen + len(labels_np)]
        seen += len(labels_np)

        for pid, y, pred, p0, p1 in zip(batch_pids, labels_np, preds, probs[:, 0], probs[:, 1]):
            rows.append(
                {
                    "patient_id": str(pid),
                    "y_true": int(y),
                    "y_pred": int(pred),
                    "prob_class0": float(p0),
                    "prob_class1": float(p1),
                }
            )

        y_true_parts.append(labels_np)
        y_pred_parts.append(preds)
        y_score_parts.append(scores)
        if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == len(loader):
            print(f"{cohort_name}: batch {batch_idx}/{len(loader)}", flush=True)

    return {
        "rows": rows,
        "y_true": np.concatenate(y_true_parts),
        "y_pred": np.concatenate(y_pred_parts),
        "y_score": np.concatenate(y_score_parts),
    }


def wilson_ci(successes, total, z=1.959963984540054):
    if total <= 0:
        return [float("nan"), float("nan")]
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt((p * (1.0 - p) / total) + (z * z / (4.0 * total * total))) / denom
    return [max(0.0, center - half), min(1.0, center + half)]


def auc_hanley_mcneil_ci(y_true, y_score, z=1.959963984540054):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    auc = float(roc_auc_score(y_true, y_score))
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return [float("nan"), float("nan")]
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (
        auc * (1.0 - auc)
        + (n_pos - 1) * (q1 - auc * auc)
        + (n_neg - 1) * (q2 - auc * auc)
    ) / (n_pos * n_neg)
    se = math.sqrt(max(var, 0.0))
    return [max(0.0, auc - z * se), min(1.0, auc + z * se)]


def balanced_acc_ci(sens, spec, sens_total, spec_total, z=1.959963984540054):
    if sens_total <= 0 or spec_total <= 0:
        return [float("nan"), float("nan")]
    var_sens = sens * (1.0 - sens) / sens_total
    var_spec = spec * (1.0 - spec) / spec_total
    se = 0.5 * math.sqrt(var_sens + var_spec)
    ba = 0.5 * (sens + spec)
    return [max(0.0, ba - z * se), min(1.0, ba + z * se)]


def compute_metrics(y_true, y_pred, y_score):
    y_true = np.asarray(y_true).astype(np.int64)
    y_pred = np.asarray(y_pred).astype(np.int64)
    y_score = np.asarray(y_score)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = int(len(y_true))
    sens_total = tp + fn
    spec_total = tn + fp
    sens = tp / sens_total if sens_total else float("nan")
    spec = tn / spec_total if spec_total else float("nan")
    acc = (tp + tn) / n if n else float("nan")
    ba = 0.5 * (sens + spec)
    ppv_total = tp + fp
    npv_total = tn + fn
    ppv = tp / ppv_total if ppv_total else float("nan")
    npv = tn / npv_total if npv_total else float("nan")
    auc = float(roc_auc_score(y_true, y_score))
    return {
        "n": n,
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "auc": auc,
        "sensitivity": sens,
        "specificity": spec,
        "accuracy": acc,
        "balanced_acc": ba,
        "ppv": ppv,
        "npv": npv,
        "ci": {
            "auc": auc_hanley_mcneil_ci(y_true, y_score),
            "sensitivity": wilson_ci(tp, sens_total),
            "specificity": wilson_ci(tn, spec_total),
            "accuracy": wilson_ci(tp + tn, n),
            "balanced_acc": balanced_acc_ci(sens, spec, sens_total, spec_total),
            "ppv": wilson_ci(tp, ppv_total),
            "npv": wilson_ci(tn, npv_total),
        },
    }


def sanitize_name(name):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")


def write_prediction_json(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def format_auc(value, ci):
    return f"{value:.3f}", f"({ci[0]:.3f}-{ci[1]:.3f})"


def format_percent(value, ci):
    return f"{value * 100.0:.1f} ({ci[0] * 100.0:.1f}-{ci[1] * 100.0:.1f})"


def table_to_csv(cohorts, path):
    rows = [
        ("AUC", "auc", False),
        ("Sensitivity (%)", "sensitivity", True),
        ("Specificity (%)", "specificity", True),
        ("Accuracy (%)", "accuracy", True),
        ("Balanced Acc. (%)", "balanced_acc", True),
        ("PPV (%)", "ppv", True),
        ("NPV (%)", "npv", True),
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Metrics"] + [f"{name} (n={data['n']})" for name, data in cohorts.items()])
        for label, key, is_percent in rows:
            values = []
            for data in cohorts.values():
                if is_percent:
                    values.append(format_percent(data[key], data["ci"][key]))
                else:
                    main, ci = format_auc(data[key], data["ci"][key])
                    values.append(f"{main} {ci}")
            writer.writerow([label] + values)


def build_html(cohorts, model_path, output_dir):
    headers = "".join(
        f"<th>{html.escape(name)}<br><span>(n={data['n']})</span></th>"
        for name, data in cohorts.items()
    )
    metric_rows = []
    for label, key, is_percent in [
        ("AUC", "auc", False),
        ("Sensitivity (%)", "sensitivity", True),
        ("Specificity (%)", "specificity", True),
        ("Accuracy (%)", "accuracy", True),
        ("Balanced Acc. (%)", "balanced_acc", True),
        ("PPV (%)", "ppv", True),
        ("NPV (%)", "npv", True),
    ]:
        cells = []
        for data in cohorts.values():
            if is_percent:
                cells.append(f"<td>{format_percent(data[key], data['ci'][key])}</td>")
            else:
                main, ci = format_auc(data[key], data["ci"][key])
                cells.append(f"<td><div class='main-val'>{main}</div><div class='ci'>{ci}</div></td>")
        metric_rows.append(f"<tr><th class='metric'>{html.escape(label)}</th>{''.join(cells)}</tr>")

    css = """
body{margin:0;padding:28px;background:#fff;color:#2b2b2b;font-family:Arial,"Microsoft YaHei","Noto Sans CJK SC",sans-serif}
.wrap{max-width:1900px;margin:0 auto}h1{font-size:30px;margin:0 0 10px;font-weight:800}
.note{color:#555;line-height:1.55;font-size:14px;margin:0 0 18px}.note code{background:#f1f3f5;padding:2px 5px;border-radius:4px}
table{width:100%;border-collapse:collapse;table-layout:fixed;border-top:4px solid #111;border-bottom:4px solid #111;font-size:26px}
th,td{border-left:2px dashed #d4d4d4;border-right:2px dashed #d4d4d4;border-bottom:2px dashed #d4d4d4;padding:12px 10px;text-align:center;vertical-align:middle;line-height:1.35}
thead th{font-size:27px;font-weight:800;border-bottom:2px solid #111}thead th span{font-size:22px}
th.metric{width:210px;font-weight:800}.main-val{font-size:27px}.ci{font-size:22px;margin-top:2px}
.files{margin-top:16px;color:#555;font-size:13px;line-height:1.6}.files code{background:#f1f3f5;padding:2px 5px;border-radius:4px;word-break:break-all}
@media print{body{padding:12px}table{font-size:18px}thead th{font-size:18px}thead th span,.ci{font-size:15px}h1{font-size:22px}}
"""
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Table 2 Diagnostic Performance</title><style>{css}</style></head><body><div class="wrap">
<h1>Table 2. Diagnostic performance of the AI model across development and validation cohorts.</h1>
<p class="note">Model: <code>{html.escape(display_path(model_path))}</code>. Values are point estimate with 95% CI. AUC CI uses Hanley-McNeil approximation; Sensitivity, Specificity, Accuracy, PPV and NPV use Wilson score intervals; Balanced Accuracy uses the delta method.</p>
<table><thead><tr><th class="metric">Metrics</th>{headers}</tr></thead><tbody>{''.join(metric_rows)}</tbody></table>
<div class="files"><div>Output directory: <code>{html.escape(display_path(output_dir))}</code></div></div>
</div></body></html>"""


def save_table_outputs(cohorts, model_path, output_dir, output_prefix):
    json_path = output_dir / f"{output_prefix}.json"
    csv_path = output_dir / f"{output_prefix}.csv"
    html_path = output_dir / f"{output_prefix}.html"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_path": display_path(model_path),
                "ci_methods": {
                    "auc": "Hanley-McNeil normal approximation",
                    "balanced_acc": "delta method from sensitivity and specificity",
                    "proportions": "Wilson score interval",
                },
                "cohorts": cohorts,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    table_to_csv(cohorts, csv_path)
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(build_html(cohorts, model_path, output_dir))
    with open(output_dir / "index.html", "w", encoding="utf-8") as f:
        f.write(f'<!doctype html><meta http-equiv="refresh" content="0; url={html.escape(html_path.name)}">')
    return html_path, csv_path, json_path


def main():
    args = parse_args()
    run_dir = args.run_dir.resolve()
    code_dir = (args.code_dir or DEFAULT_CODE_DIR).resolve()
    model_path = run_dir / args.model_name
    data_config_path = run_dir / "data_config.json"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not model_path.exists():
        raise FileNotFoundError(model_path)
    if not data_config_path.exists():
        raise FileNotFoundError(data_config_path)

    dataset_module = load_module("table2_dataset", code_dir / "dataset.py")
    model_module = load_module("table2_model", code_dir / "model.py")
    dataset_cls = dataset_module.ThymomaDataset
    model_cls = model_module.ThymomaTransformerClassifier

    data_config = load_json(data_config_path)
    def config_path(key):
        path = Path(data_config[key])
        return path if path.is_absolute() else PACKAGE_ROOT / path

    image_size = int(data_config.get("image_size", 224))
    max_slices = int(data_config.get("max_slices", data_config.get("target_slices", 32)))
    device = torch.device(args.device)
    model, checkpoint, num_layers = load_model(model_cls, model_path, device, max_slices, args.drop_out)

    eval_specs = [
        {
            "name": "Training Cohort",
            "image_root": config_path("data_root"),
            "json_path": config_path("train_json"),
            "save_name": "training_cohort",
        },
        {
            "name": "Internal Validation",
            "image_root": config_path("data_root"),
            "json_path": config_path("val_json"),
            "save_name": "internal_validation",
        },
    ]
    external_specs = []
    for idx, json_name in enumerate(args.external_eval_sets, start=1):
        external_specs.append(
            {
                "name": f"External Center {idx}",
                "image_root": args.external_image_root,
                "json_path": args.external_json_root / json_name,
                "save_name": Path(json_name).stem,
            }
        )
    eval_specs.extend(external_specs)
    eval_specs.append(
        {
            "name": "Prospective Cohort",
            "image_root": args.prospective_image_root,
            "json_path": args.prospective_json,
            "save_name": "prospective",
        }
    )

    cohorts = {}
    external_true = []
    external_pred = []
    external_score = []
    meta = {
        "run_dir": display_path(run_dir),
        "code_dir": display_path(code_dir),
        "model_path": display_path(model_path),
        "checkpoint_epoch": checkpoint.get("epoch") if isinstance(checkpoint, dict) else None,
        "checkpoint_metrics": {
            key: checkpoint[key]
            for key in ("val_acc", "val_auc")
            if isinstance(checkpoint, dict) and key in checkpoint
        },
        "num_layers": num_layers,
        "max_slices": max_slices,
        "image_size": image_size,
        "drop_out": args.drop_out,
        "threshold": args.threshold,
        "external_json_root": display_path(args.external_json_root),
        "external_image_root": display_path(args.external_image_root),
        "external_eval_sets": args.external_eval_sets,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "device": str(device),
    }

    for spec in eval_specs:
        print(f"Running {spec['name']} from {spec['json_path']}", flush=True)
        result = predict_dataset(
            model=model,
            dataset_cls=dataset_cls,
            cohort_name=spec["name"],
            image_root=spec["image_root"],
            json_path=spec["json_path"],
            image_size=image_size,
            max_slices=max_slices,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device=device,
            threshold=args.threshold,
        )
        prediction_path = output_dir / f"{sanitize_name(spec['save_name'])}_predictions.json"
        write_prediction_json(prediction_path, result["rows"])
        cohorts[spec["name"]] = compute_metrics(result["y_true"], result["y_pred"], result["y_score"])
        cohorts[spec["name"]]["prediction_path"] = display_path(prediction_path)

        if spec in external_specs:
            external_true.append(result["y_true"])
            external_pred.append(result["y_pred"])
            external_score.append(result["y_score"])

    external_combined = compute_metrics(
        np.concatenate(external_true),
        np.concatenate(external_pred),
        np.concatenate(external_score),
    )
    external_combined["prediction_path"] = "combined from external center prediction files"

    ordered = {}
    for name, data in cohorts.items():
        if name == "Prospective Cohort":
            ordered["External Combined"] = external_combined
        ordered[name] = data
    cohorts = ordered

    with open(output_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    html_path, csv_path, json_path = save_table_outputs(
        cohorts,
        model_path,
        output_dir,
        args.output_prefix,
    )
    print(f"Saved HTML: {html_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
