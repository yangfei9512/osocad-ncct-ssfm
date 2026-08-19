#!/usr/bin/env python3

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader


PACKAGE_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_PATH = PACKAGE_ROOT / "models" / "best.pth"
DEFAULT_RUN_DIR = PACKAGE_ROOT / "models"
DEFAULT_CODE_DIR = PACKAGE_ROOT / "code" / "training"
DEFAULT_OUTPUT_XLSX = PACKAGE_ROOT / "metrics" / "table2_inference_all_cohorts.xlsx"
DEFAULT_EXTERNAL_JSON_ROOT = PACKAGE_ROOT / "data" / "external" / "json"
DEFAULT_EXTERNAL_IMAGE_ROOT = PACKAGE_ROOT / "data" / "external" / "images"
DEFAULT_PROSPECTIVE_ROOT = PACKAGE_ROOT / "data" / "prospective"


def display_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(PACKAGE_ROOT))
    except ValueError:
        return str(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    parser.add_argument("--code-dir", type=Path, default=DEFAULT_CODE_DIR)
    parser.add_argument("--output-xlsx", type=Path, default=DEFAULT_OUTPUT_XLSX)
    parser.add_argument("--external-json-root", type=Path, default=DEFAULT_EXTERNAL_JSON_ROOT)
    parser.add_argument("--external-image-root", type=Path, default=DEFAULT_EXTERNAL_IMAGE_ROOT)
    parser.add_argument("--prospective-root", type=Path, default=DEFAULT_PROSPECTIVE_ROOT)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--drop-out", type=float, default=0.1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--include-center3", action="store_true")
    return parser.parse_args()


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def infer_num_layers(state_dict: dict[str, torch.Tensor]) -> int:
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


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_model(model_cls, model_path: Path, device: torch.device, max_slices: int, drop_out: float):
    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    num_layers = infer_num_layers(state_dict)
    if any(key.startswith("slice_encoder.backbone.stem_") or key.startswith("slice_encoder.backbone.stages_")
           for key in state_dict):
        backbone_name = "convnext_tiny"
    else:
        backbone_name = "vit_tiny_patch16_224"
    model = model_cls(
        backbone_name=backbone_name,
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
def predict_cohort(
    model,
    dataset_cls,
    cohort_name: str,
    image_root: Path,
    json_path: Path,
    image_size: int,
    max_slices: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    threshold: float,
) -> pd.DataFrame:
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
    rows = []
    seen = 0
    patient_ids = dataset.patient_ids
    for batch_idx, (imgs, labels) in enumerate(loader, start=1):
        imgs = imgs.to(device, non_blocking=True)
        logits = model(imgs)
        probs = F.softmax(logits, dim=1).detach().cpu().numpy()
        scores = probs[:, 1]
        preds = (scores >= threshold).astype(np.int64)
        labels_np = labels.numpy().astype(np.int64)
        batch_pids = patient_ids[seen : seen + len(labels_np)]
        seen += len(labels_np)
        for pid, y_true, y_pred, p0, p1 in zip(batch_pids, labels_np, preds, probs[:, 0], probs[:, 1]):
            rows.append(
                {
                    "cohort": cohort_name,
                    "patient_id": str(pid),
                    "y_true": int(y_true),
                    "prob_class0": float(p0),
                    "prob_class1": float(p1),
                    "y_pred": int(y_pred),
                    "correct": int(y_true == y_pred),
                }
            )
        if batch_idx == 1 or batch_idx % 10 == 0 or batch_idx == len(loader):
            print(f"{cohort_name}: batch {batch_idx}/{len(loader)}", flush=True)
    return pd.DataFrame(rows)


def wilson_ci(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return (float("nan"), float("nan"))
    p = successes / total
    denom = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / denom
    half = z * math.sqrt((p * (1.0 - p) / total) + (z * z / (4.0 * total * total))) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def auc_hanley_mcneil_ci(y_true: np.ndarray, y_score: np.ndarray, z: float = 1.959963984540054) -> tuple[float, float]:
    auc = float(roc_auc_score(y_true, y_score))
    n_pos = int(np.sum(y_true == 1))
    n_neg = int(np.sum(y_true == 0))
    if n_pos == 0 or n_neg == 0:
        return (float("nan"), float("nan"))
    q1 = auc / (2.0 - auc)
    q2 = 2.0 * auc * auc / (1.0 + auc)
    var = (
        auc * (1.0 - auc)
        + (n_pos - 1) * (q1 - auc * auc)
        + (n_neg - 1) * (q2 - auc * auc)
    ) / (n_pos * n_neg)
    se = math.sqrt(max(var, 0.0))
    return (max(0.0, auc - z * se), min(1.0, auc + z * se))


def balanced_acc_ci(sens: float, spec: float, sens_total: int, spec_total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if sens_total <= 0 or spec_total <= 0:
        return (float("nan"), float("nan"))
    var_sens = sens * (1.0 - sens) / sens_total
    var_spec = spec * (1.0 - spec) / spec_total
    se = 0.5 * math.sqrt(var_sens + var_spec)
    ba = 0.5 * (sens + spec)
    return (max(0.0, ba - z * se), min(1.0, ba + z * se))


def compute_metrics(df: pd.DataFrame) -> dict[str, object]:
    y_true = df["y_true"].to_numpy(dtype=np.int64)
    y_pred = df["y_pred"].to_numpy(dtype=np.int64)
    y_score = df["prob_class1"].to_numpy(dtype=float)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    n = int(len(df))
    sens_total = int(tp + fn)
    spec_total = int(tn + fp)
    ppv_total = int(tp + fp)
    npv_total = int(tn + fn)
    sens = tp / sens_total if sens_total else float("nan")
    spec = tn / spec_total if spec_total else float("nan")
    acc = (tp + tn) / n if n else float("nan")
    ba = 0.5 * (sens + spec)
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
        "ci_auc_low": auc_hanley_mcneil_ci(y_true, y_score)[0],
        "ci_auc_high": auc_hanley_mcneil_ci(y_true, y_score)[1],
        "ci_sensitivity_low": wilson_ci(int(tp), sens_total)[0],
        "ci_sensitivity_high": wilson_ci(int(tp), sens_total)[1],
        "ci_specificity_low": wilson_ci(int(tn), spec_total)[0],
        "ci_specificity_high": wilson_ci(int(tn), spec_total)[1],
        "ci_accuracy_low": wilson_ci(int(tp + tn), n)[0],
        "ci_accuracy_high": wilson_ci(int(tp + tn), n)[1],
        "ci_balanced_acc_low": balanced_acc_ci(sens, spec, sens_total, spec_total)[0],
        "ci_balanced_acc_high": balanced_acc_ci(sens, spec, sens_total, spec_total)[1],
        "ci_ppv_low": wilson_ci(int(tp), ppv_total)[0],
        "ci_ppv_high": wilson_ci(int(tp), ppv_total)[1],
        "ci_npv_low": wilson_ci(int(tn), npv_total)[0],
        "ci_npv_high": wilson_ci(int(tn), npv_total)[1],
    }


def fmt_auc(value: float, low: float, high: float) -> str:
    return f"{value:.3f}\n({low:.3f}-{high:.3f})"


def fmt_pct(value: float, low: float, high: float) -> str:
    return f"{value * 100:.1f} ({low * 100:.1f}-{high * 100:.1f})"


def make_table2_sheet(metrics_by_name: OrderedDict[str, dict[str, object]]) -> pd.DataFrame:
    metric_specs = [
        ("AUC", "auc", False),
        ("Sensitivity (%)", "sensitivity", True),
        ("Specificity (%)", "specificity", True),
        ("Accuracy (%)", "accuracy", True),
        ("Balanced Acc. (%)", "balanced_acc", True),
        ("PPV (%)", "ppv", True),
        ("NPV (%)", "npv", True),
    ]
    rows = []
    for label, key, is_percent in metric_specs:
        row = {"Metrics": label}
        for cohort, metric in metrics_by_name.items():
            if is_percent:
                row[f"{cohort}\n(n={metric['n']})"] = fmt_pct(
                    metric[key],
                    metric[f"ci_{key}_low"],
                    metric[f"ci_{key}_high"],
                )
            else:
                row[f"{cohort}\n(n={metric['n']})"] = fmt_auc(
                    metric[key],
                    metric["ci_auc_low"],
                    metric["ci_auc_high"],
                )
        rows.append(row)
    return pd.DataFrame(rows)


def safe_sheet_name(name: str, used: set[str]) -> str:
    cleaned = re.sub(r"[:\\/?*\[\]]", "_", name)[:31]
    base = cleaned or "sheet"
    candidate = base
    idx = 1
    while candidate in used:
        suffix = f"_{idx}"
        candidate = base[: 31 - len(suffix)] + suffix
        idx += 1
    used.add(candidate)
    return candidate


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    code_dir = args.code_dir.resolve()
    model_path = args.model_path.resolve()
    output_xlsx = args.output_xlsx.resolve()
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)

    data_config = load_json(run_dir / "data_config.json")
    def config_path(key: str) -> Path:
        path = Path(data_config[key])
        return path if path.is_absolute() else PACKAGE_ROOT / path

    image_size = int(data_config.get("image_size", 224))
    max_slices = int(data_config.get("target_slices", data_config.get("max_slices", 32)))
    device = torch.device(args.device)

    dataset_module = load_module("table2_dataset_export", code_dir / "dataset.py")
    model_module = load_module("table2_model_export", code_dir / "model.py")
    dataset_cls = dataset_module.ThymomaDataset
    model_cls = model_module.ThymomaTransformerClassifier
    model, checkpoint, num_layers = load_model(model_cls, model_path, device, max_slices, args.drop_out)

    external_json_root = args.external_json_root.resolve()
    external_image_root = args.external_image_root.resolve()
    cohort_specs = [
        ("Training Cohort", config_path("data_root"), config_path("train_json")),
        ("Internal Validation", config_path("data_root"), config_path("val_json")),
        ("External Center 1", external_image_root, external_json_root / "external_shuguang.json"),
        ("External Center 2", external_image_root, external_json_root / "external_huangshan.json"),
    ]
    if args.include_center3:
        cohort_specs.append(
            ("External Center 3", external_image_root, external_json_root / "external_hospital5_included_in_train_test.json")
        )
    cohort_specs.append(
        (
            "Prospective Cohort 410",
            args.prospective_root / "images",
            args.prospective_root / "json" / "forward_ct.json",
        )
    )
    predictions: OrderedDict[str, pd.DataFrame] = OrderedDict()
    failures = []
    for name, image_root, json_path in cohort_specs:
        print(f"Running {name}: {json_path}", flush=True)
        try:
            predictions[name] = predict_cohort(
                model=model,
                dataset_cls=dataset_cls,
                cohort_name=name,
                image_root=image_root,
                json_path=json_path,
                image_size=image_size,
                max_slices=max_slices,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                threshold=args.threshold,
            )
        except Exception as exc:
            failures.append(
                {
                    "cohort": name,
                    "image_root": display_path(image_root),
                    "json_path": display_path(json_path),
                    "error": repr(exc),
                }
            )
            print(f"[WARN] skipped {name}: {exc}", flush=True)

    external_frames = [
        predictions[name]
        for name in ("External Center 1", "External Center 2")
        if name in predictions
    ]
    if external_frames:
        external_combined = pd.concat(external_frames, ignore_index=True)
        external_combined["cohort"] = "External Combined"
        predictions["External Combined"] = external_combined

    table_order = [
        "Training Cohort",
        "Internal Validation",
        "External Center 1",
        "External Center 2",
        "External Combined",
        "Prospective Cohort 410",
    ]
    if args.include_center3:
        table_order.insert(4, "External Center 3")
    table_order = [name for name in table_order if name in predictions]
    metrics_by_name = OrderedDict((name, compute_metrics(predictions[name])) for name in table_order)
    metrics_raw = pd.DataFrame([{"cohort": name, **metric} for name, metric in metrics_by_name.items()])
    table2 = make_table2_sheet(metrics_by_name)
    all_cases = pd.concat([predictions[name] for name in table_order if name != "External Combined"], ignore_index=True)

    meta = pd.DataFrame(
        [
            {"key": "model_path", "value": display_path(model_path)},
            {"key": "run_dir", "value": display_path(run_dir)},
            {"key": "code_dir", "value": display_path(code_dir)},
            {"key": "checkpoint_epoch", "value": checkpoint.get("epoch") if isinstance(checkpoint, dict) else ""},
            {"key": "checkpoint_val_auc", "value": checkpoint.get("val_auc") if isinstance(checkpoint, dict) else ""},
            {"key": "checkpoint_val_acc", "value": checkpoint.get("val_acc") if isinstance(checkpoint, dict) else ""},
            {"key": "num_layers", "value": num_layers},
            {"key": "max_slices", "value": max_slices},
            {"key": "image_size", "value": image_size},
            {"key": "threshold", "value": args.threshold},
            {"key": "auc_ci", "value": "Hanley-McNeil normal approximation"},
            {"key": "proportion_ci", "value": "Wilson score interval"},
            {"key": "balanced_acc_ci", "value": "Delta method from sensitivity and specificity"},
            {"key": "failed_cohorts", "value": json.dumps(failures, ensure_ascii=False)},
        ]
    )

    with pd.ExcelWriter(output_xlsx, engine="openpyxl") as writer:
        table2.to_excel(writer, sheet_name="Table2_summary", index=False)
        metrics_raw.to_excel(writer, sheet_name="Metrics_raw", index=False)
        meta.to_excel(writer, sheet_name="Meta", index=False)
        pd.DataFrame(failures).to_excel(writer, sheet_name="Failures", index=False)
        all_cases.to_excel(writer, sheet_name="All_cases", index=False)
        used = {"Table2_summary", "Metrics_raw", "Meta", "Failures", "All_cases"}
        for name in table_order:
            sheet = safe_sheet_name(name, used)
            predictions[name].to_excel(writer, sheet_name=sheet, index=False)

        ws = writer.book["Table2_summary"]
        for row in ws.iter_rows():
            for cell in row:
                cell.alignment = cell.alignment.copy(wrap_text=True, horizontal="center", vertical="center")
        for column_cells in ws.columns:
            ws.column_dimensions[column_cells[0].column_letter].width = 24
        ws.column_dimensions["A"].width = 20

    metrics_raw.to_csv(output_xlsx.with_suffix(".metrics_raw.csv"), index=False, encoding="utf-8-sig")
    all_cases.to_csv(output_xlsx.with_suffix(".all_cases.csv"), index=False, encoding="utf-8-sig")
    print(f"Saved Excel: {output_xlsx}", flush=True)


if __name__ == "__main__":
    main()
