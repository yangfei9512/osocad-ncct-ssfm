# Opportunistic screening of obstructive coronary artery disease from non-contrast chest CT by a self-supervised foundation model

## Task

Coronary CT binary classification, including model development, internal cohort evaluation, external cohort evaluation, prospective cohort evaluation, paper figure export, and calcium-score comparison.

## Repository Layout

- `code/training/`: dataset, model, training launcher, and PNG preprocessing code.
- `code/inference/`: checkpoint inference and validation entry points.
- `code/metrics/`: diagnostic-performance and Table 2 metric code.
- `code/plotting/`: manuscript table export and figure-redraw utilities.
- `models/best.pth`: packaged model checkpoint.
- `paper_plots/`: final result tables and figures.
- `20260624T115247_final_all_tables_figures.xlsx`: final all-in-one workbook containing tables, case-level results, source metrics, p values, and embedded figures.

## Data Layout

Place the data under the repository root using the following structure. Each label JSON maps a patient ID to a binary label (`0` or `1`), and the corresponding image directory contains that patient's PNG/JPG slices.

```text
data/
├── internal/
│   ├── images/<patient_id>/*.{png,jpg,jpeg}
│   └── json/{train.json,validation.json}
├── external/
│   ├── images/<patient_id>/*.{png,jpg,jpeg}
│   └── json/{external_shuguang.json,external_huangshan.json}
└── prospective/
    ├── images/<patient_id>/*.{png,jpg,jpeg}
    └── json/forward_ct.json
```

Place the pretrained ViT backbone at `models/pytorch_model.bin`. The packaged classifier checkpoint and its data configuration are `models/best.pth` and `models/data_config.json`.

## Usage

Run the training grid:

```bash
python code/training/launch.py
```

To use an internal dataset stored elsewhere, set `INTERNAL_DATASET_ROOT`:

```bash
INTERNAL_DATASET_ROOT=datasets/internal python code/training/launch.py
```

### Training Settings Used

| Setting | Value |
| --- | --- |
| Backbone | ViT-Tiny (`vit_tiny_patch16_224`) |
| Input | Heart-ROI PNG slices, WL/WW 40/400 |
| Image size | 224 × 224 |
| Target/max slices | 32 / 32 |
| Batch size | 16 |
| Loss | Focal loss, gamma = 2.0 |
| Optimizer | AdamW |
| Weight decay | 0.1 |
| Scheduler | Cosine annealing |
| Learning rates | `1e-5`, `2e-5` |
| Transformer layers | `4`, `6`, `8`, `16` |
| Dropout | 0.2 |
| Epochs | 1000 |
| Data workers | 4 |
| GPUs per run | 1 |

Run one training configuration:

```bash
torchrun --nproc_per_node=1 --master_port=29501 code/training/train.py \
  --CHECKPOINT_PATH runs/example_run \
  --data_root data/internal/images \
  --train_json data/internal/json/train.json \
  --val_json data/internal/json/validation.json \
  --batch_size 16 --lr 2e-5 --drop_out 0.2 --weight_decay 0.1 \
  --image_size 224 --max_slices 32 --target_slices 32 \
  --num_layers 16 --num_epochs 1000 --num_workers 4 \
  --loss focal --focal_gamma 2.0
```

Evaluate the packaged model on the two external centers:

```bash
python code/inference/eval_external.py \
  --model-path models/best.pth \
  --result-dir metrics/external_evaluation
```

Generate all-cohort metrics, prediction files, HTML/CSV/JSON summaries, and the Excel table:

```bash
python code/inference/infer_table2_diagnostic_performance.py
python code/metrics/export_checkpoint_table2_excel.py
```

Regenerate the calcium-score comparison figures and refresh the workbook/HTML outputs:

```bash
python code/plotting/redraw_calcium_sens_npv_from_excel.py
```

## Table 3 Metrics

| Metrics | Proposed AI Model (NCCT) | Non-gated Agatston Score (NCCT) | Gated Agatston Score (Dedicated CSCT) | Comparison with Non-gated Score | Comparison with Gated Score |
| --- | --- | --- | --- | --- | --- |
| External Cohorts |  |  |  |  |  |
| AUC | 0.848 (0.817 - 0.874) | 0.824 (0.794 - 0.853) | 0.868 (0.842 - 0.894) | p = 0.045088 (p < 0.05) | p = 0.0457172 (p < 0.05) |
| Sensitivity (%) | 81.8 (77.6 - 85.6) | 40.6 (35.4 - 45.9) | 57.9 (52.9 - 63.8) | p = 1.71235e-37 (p < 0.05) | p = 3.01885e-20 (p < 0.05) |
| Specificity (%) | 74.1 (69.5 - 78.4) | 97.1 (95.5 - 98.6) | 92.3 (89.7 - 94.8) | p = 2.52435e-29 (p < 0.05) | p = 5.36178e-21 (p < 0.05) |
| NPV (%) | 83.7 (80.3 - 87.2) | 67.4 (64.1 - 71.0) | 73.5 (70.2 - 77.3) | Delta +16.4 pp (+13.0 to +19.7); p = 9.999e-05 (p < 0.05) | Delta +10.3 pp (+7.2 to +13.3); p = 9.999e-05 (p < 0.05) |
| Prospective Cohort |  |  |  |  |  |
| AUC | 0.886 (0.841 - 0.926) | 0.857 (0.814 - 0.901) | 0.898 (0.858 - 0.934) | p = 0.0727101 (ns) | p = 0.321406 (ns) |
| Sensitivity (%) | 84.3 (77.1 - 91.1) | 60.2 (51.0 - 69.9) | 75.9 (67.6 - 83.5) | p = 2.98023e-08 (p < 0.05) | p = 0.00390625 (p < 0.05) |
| Specificity (%) | 75.5 (70.5 - 80.3) | 94.4 (91.5 - 96.8) | 91.7 (88.8 - 94.9) | p = 3.91866e-14 (p < 0.05) | p = 3.17968e-13 (p < 0.05) |
| NPV (%) | 93.1 (89.8 - 96.1) | 86.9 (83.2 - 90.6) | 91.4 (88.3 - 94.4) | Delta +6.2 pp (+3.4 to +9.2); p = 0.00019998 (p < 0.05) | Delta +1.6 pp (-0.1 to +3.7); p = 0.0911909 (ns) |

## Result Files

- `paper_plots/table1_ai_performance.csv`
- `paper_plots/table2_calcium_comparison.csv`

## Figures

- `paper_plots/Fig2_A.png`
- `paper_plots/Fig2_B.png`
- `paper_plots/Fig2_C.png`
- `paper_plots/Fig3_A.png`
- `paper_plots/Fig3_B.png`
- `paper_plots/Fig3_C.png`
- `paper_plots/Fig3_D.png`
- `paper_plots/Fig4_A.png`
- `paper_plots/Fig4_B.png`
- `paper_plots/Fig4_C.png`
- `paper_plots/Fig4_D.png`
- `paper_plots/Fig5_A.png`
- `paper_plots/Fig5_B.png`
- `paper_plots/Fig5_C.png`
- `paper_plots/Fig5_D.png`
