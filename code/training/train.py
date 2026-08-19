import os
import argparse
import json
from pathlib import Path
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-sh-rj")
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from dataset import ThymomaDataset
from model import ThymomaTransformerClassifier
from torch.optim.lr_scheduler import CosineAnnealingLR

PACKAGE_ROOT = Path(__file__).resolve().parents[2]


def resolve_repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else PACKAGE_ROOT / path

# --------------------------
# FocalLoss (optional)
# --------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = torch.tensor(alpha, dtype=torch.float32) if alpha is not None else None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        probs = F.softmax(inputs, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        at = self.alpha[targets].to(inputs.device) if self.alpha is not None else 1.0
        loss = at * ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

# --------------------------
# DDP setup
# --------------------------
def setup_ddp():
    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return torch.device("cuda", local_rank), local_rank

# --------------------------
# Argument parser
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--CHECKPOINT_PATH", type=str, required=True)
    parser.add_argument("--data_root", type=str, required=True)
    parser.add_argument("--train_json", type=str, required=True)
    parser.add_argument("--val_json", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--drop_out", type=float, default=0.2)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--max_slices", type=int, default=64)
    parser.add_argument("--target_slices", type=int, default=None)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--loss", type=str, default="focal", choices=["ce", "focal"])
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    return parser.parse_args()

# --------------------------
# Evaluate helpers
# --------------------------
def compute_binary_metrics(labels_np, probs):
    preds = np.argmax(probs, axis=1)
    acc = accuracy_score(labels_np, preds)
    num_classes = probs.shape[1]
    auc_per_class = {}
    valid_classes = []
    for i in range(num_classes):
        binary_labels = (labels_np == i).astype(int)
        if np.sum(binary_labels) == 0:
            auc_per_class[i] = None
        else:
            auc_i = roc_auc_score(binary_labels, probs[:, i])
            auc_per_class[i] = auc_i
            valid_classes.append(i)
    macro_auc = np.mean([auc_per_class[i] for i in valid_classes]) if valid_classes else 0.0

    cm = confusion_matrix(labels_np, preds, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
    else:
        tn = fp = fn = tp = 0
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "acc": float(acc),
        "macro_auc": float(macro_auc),
        "sens": float(sens),
        "spec": float(spec),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "preds": preds,
        "auc_per_class": auc_per_class,
    }


@torch.no_grad()
def evaluate(mode, model, dataloader, device, epoch=None, is_main=False, save_path=None):
    model.eval()
    all_logits, all_targets = [], []

    for imgs, labels in dataloader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        all_logits.append(logits.cpu())
        all_targets.append(labels.cpu())

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_targets, dim=0)
    probs = F.softmax(logits, dim=1).numpy()
    labels_np = labels.numpy()
    metrics = compute_binary_metrics(labels_np, probs)
    preds = metrics["preds"]
    auc_per_class = metrics["auc_per_class"]

    if is_main and save_path:
        os.makedirs(save_path, exist_ok=True)
        # ROC & Confusion Matrix
        for cls_id in range(probs.shape[1]):
            if auc_per_class.get(cls_id) is None:
                continue
            fpr, tpr, _ = roc_curve((labels_np == cls_id).astype(int), probs[:, cls_id])
            plt.figure(figsize=(6,6))
            plt.plot(fpr, tpr, label=f"AUC={auc_per_class[cls_id]:.4f}")
            plt.plot([0,1],[0,1],'k--')
            plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"{mode.upper()} ROC Class {cls_id}")
            plt.legend(); plt.grid(True)
            plt.savefig(os.path.join(save_path, f"{mode}_roc_class{cls_id}_epoch{epoch}.png"), dpi=300)
            plt.close()

        cm = confusion_matrix(labels_np, preds)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap=plt.cm.Blues)
        plt.title(f"{mode.upper()} Confusion Matrix")
        plt.savefig(os.path.join(save_path, f"{mode}_confusion_epoch{epoch}.png"), dpi=300)
        plt.close()

    # Print metrics
    print(
        f"[{mode.upper()} Epoch {epoch}] ACC: {metrics['acc']:.4f}, Macro AUC: {metrics['macro_auc']:.4f}, "
        f"Sens: {metrics['sens']:.4f}, Spec: {metrics['spec']:.4f}, "
        f"TN: {metrics['tn']}, FP: {metrics['fp']}, FN: {metrics['fn']}, TP: {metrics['tp']}"
    )

    return metrics

# --------------------------
# Main training
# --------------------------
def main():
    args = parse_args()
    target_slices = args.target_slices if args.target_slices is not None else args.max_slices
    device, local_rank = setup_ddp()
    is_main = dist.get_rank() == 0
    if is_main:
        os.makedirs(args.CHECKPOINT_PATH, exist_ok=True)
        writer = SummaryWriter(log_dir=args.CHECKPOINT_PATH)
        data_config = {
            "data_root": args.data_root,
            "train_json": args.train_json,
            "val_json": args.val_json,
            "max_slices": args.max_slices,
            "target_slices": target_slices,
            "image_size": args.image_size,
            "num_workers": args.num_workers,
        }
        with open(os.path.join(args.CHECKPOINT_PATH, "data_config.json"), "w", encoding="utf-8") as f:
            json.dump(data_config, f, ensure_ascii=False, indent=2)
        print("Data config:", json.dumps(data_config, ensure_ascii=False))

    # Datasets
    data_root = resolve_repo_path(args.data_root)
    train_json = resolve_repo_path(args.train_json)
    val_json = resolve_repo_path(args.val_json)
    train_dataset = ThymomaDataset(image_root=data_root, label_json=train_json, is_train=True,
                                   max_slices=args.max_slices, target_slices=target_slices,
                                   image_size=args.image_size)
    val_dataset = ThymomaDataset(image_root=data_root, label_json=val_json, is_train=False,
                                 max_slices=args.max_slices, target_slices=target_slices,
                                 image_size=args.image_size)
    patient_overlap = set(train_dataset.patient_ids).intersection(val_dataset.patient_ids)
    if patient_overlap:
        raise ValueError(
            "Training and internal-validation sets contain overlapping patient ids: "
            f"{sorted(patient_overlap)[:5]}"
        )

    train_sampler = DistributedSampler(train_dataset)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    train_loader_kwargs = {"num_workers": args.num_workers}
    eval_loader_kwargs = {"num_workers": min(1, args.num_workers)}
    if args.num_workers > 0:
        train_loader_kwargs["prefetch_factor"] = 1
        train_loader_kwargs["persistent_workers"] = True
    if eval_loader_kwargs["num_workers"] > 0:
        eval_loader_kwargs["prefetch_factor"] = 1
        eval_loader_kwargs["persistent_workers"] = True

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, **train_loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, **eval_loader_kwargs)

    # Model
    model = ThymomaTransformerClassifier(
        backbone_name='vit_tiny_patch16_224',
        num_classes=2,
        max_slices=args.max_slices,
        drop_out=args.drop_out,
        num_layers=args.num_layers
    ).to(device)
    model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-10)
    if args.loss == "focal":
        cls_loss_function = FocalLoss(gamma=args.focal_gamma)
    else:
        cls_loss_function = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_val_auc = 0.0
    best_val_acc = 0.0

    for epoch in range(args.num_epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        total_loss = 0.0
        all_logits_train, all_labels_train = [], []
        optimizer.zero_grad()

        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits = model(imgs)
            loss = cls_loss_function(logits, labels)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

            all_logits_train.append(logits.detach().cpu())
            all_labels_train.append(labels.detach().cpu())

        # Training metrics
        logits_train = torch.cat(all_logits_train, dim=0)
        labels_train = torch.cat(all_labels_train, dim=0)
        probs_train = F.softmax(logits_train, dim=1).numpy()
        preds_train = np.argmax(probs_train, axis=1)
        labels_np_train = labels_train.numpy()
        acc_train = accuracy_score(labels_np_train, preds_train)
        num_classes = probs_train.shape[1]
        auc_per_class_train = []
        for i in range(num_classes):
            binary_labels = (labels_np_train == i).astype(int)
            if np.sum(binary_labels) > 0:
                auc_per_class_train.append(roc_auc_score(binary_labels, probs_train[:, i]))
        macro_auc_train = np.mean(auc_per_class_train) if auc_per_class_train else 0.0

        if is_main:
            writer.add_scalar("Loss/Train_epoch", total_loss / len(train_loader), epoch)
            writer.add_scalar("AUC/Train", macro_auc_train, epoch)
            writer.add_scalar("ACC/Train", acc_train, epoch)
            print(f"[Epoch {epoch}] Train Loss: {total_loss / len(train_loader):.4f}, "
                  f"Train ACC: {acc_train:.4f}, Train AUC: {macro_auc_train:.4f}")

        scheduler.step()

        # Validation
        val_metrics = evaluate("val", model, val_loader, device, epoch, is_main, args.CHECKPOINT_PATH)
        if is_main:
            writer.add_scalar("AUC/Val", val_metrics["macro_auc"], epoch)
            writer.add_scalar("ACC/Val", val_metrics["acc"], epoch)
            writer.add_scalar("Sens/Val", val_metrics["sens"], epoch)
            writer.add_scalar("Spec/Val", val_metrics["spec"], epoch)

            # Save the final model when internal-validation AUC improves.
            if val_metrics["macro_auc"] > best_val_auc:
                best_val_auc = val_metrics["macro_auc"]
                state = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'val_auc': val_metrics["macro_auc"],
                    'val_acc': val_metrics["acc"],
                }
                torch.save(state, os.path.join(args.CHECKPOINT_PATH, "best_model_auc.pth"))
                torch.save(state, os.path.join(args.CHECKPOINT_PATH, "best.pth"))
                print(
                    f"Saved best internal-validation AUC model at epoch {epoch} "
                    f"with AUC {best_val_auc:.4f}"
                )

            # Save model when validation ACC improves
            if val_metrics["acc"] > best_val_acc:
                best_val_acc = val_metrics["acc"]
                state = {
                    'epoch': epoch,
                    'model_state_dict': model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
                    'val_auc': val_metrics["macro_auc"],
                    'val_acc': val_metrics["acc"],
                }
                torch.save(state, os.path.join(args.CHECKPOINT_PATH, "best_model_acc.pth"))
                print(f"Saved best ACC model at epoch {epoch} with ACC {best_val_acc:.4f}")

        dist.barrier()

    if is_main:
        writer.close()
    dist.destroy_process_group()

if __name__ == "__main__":
    main()
