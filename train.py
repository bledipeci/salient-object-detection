"""
train.py
The full training loop — forward pass, backward pass, validation, and checkpointing.

A few design decisions worth noting:
- We use AdamW instead of Adam because it handles weight decay more correctly
  for adaptive optimisers (Adam's built-in weight_decay is actually L2 regularisation,
  which gets scaled by the gradient magnitude — AdamW keeps them separate).
- The LR schedule does a short warmup then cosine decay. Warmup prevents large
  gradients from destabilising BatchNorm statistics in the first few epochs.
- Gradient clipping stops the occasional exploding gradient from wrecking a run.
- Early stopping watches val IoU (not val loss) because our combined loss includes
  SSIM which can sometimes increase even when the actual predictions improve.
- We save two checkpoints per run: 'last' for resume-ability, 'best' for evaluation.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from data_loader import get_dataloaders
from sod_model import CombinedLoss, build_model, count_parameters


def _batch_iou(pred, target, threshold=0.5, eps=1e-6):
    """Quick IoU estimate for a single batch, used for progress logging."""
    pred_bin = (pred > threshold).float()
    inter    = (pred_bin * target).sum(dim=(1, 2, 3))
    union    = pred_bin.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3)) - inter
    return ((inter + eps) / (union + eps)).mean().item()


def _cosine_with_warmup(optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
    """
    Linear warmup for the first `warmup_epochs`, then cosine decay to `eta_min`.
    This is implemented as a LambdaLR multiplier on the base learning rate.
    """
    for pg in optimizer.param_groups:
        pg["initial_lr"] = pg["lr"]  # save original LR so the lambda can reference it

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
        base_lr  = optimizer.param_groups[0]["initial_lr"]
        ratio    = eta_min / base_lr if base_lr > 0 else 0.0
        return ratio + (1.0 - ratio) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, loss_fn, device, scaler, clip_norm):
    model.train()
    total_loss, total_iou = 0.0, 0.0

    pbar = tqdm(loader, desc="  train", leave=False, unit="batch")
    for images, masks in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()

        # AMP (automatic mixed precision) cuts memory and speeds up training on GPU
        with autocast("cuda", enabled=(scaler is not None)):
            preds = model(images)
            loss  = loss_fn(preds, masks)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
            optimizer.step()

        iou = _batch_iou(preds.detach(), masks)
        total_loss += loss.item()
        total_iou  += iou
        pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou:.4f}")

    n = len(loader)
    return total_loss / n, total_iou / n


def validate(model, loader, loss_fn, device):
    model.eval()
    total_loss, total_iou = 0.0, 0.0

    with torch.no_grad():
        pbar = tqdm(loader, desc="  val  ", leave=False, unit="batch")
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            loss  = loss_fn(preds, masks)
            iou   = _batch_iou(preds, masks)
            total_loss += loss.item()
            total_iou  += iou
            pbar.set_postfix(loss=f"{loss.item():.4f}", iou=f"{iou:.4f}")

    n = len(loader)
    return total_loss / n, total_iou / n


def _empty_history():
    return {"train_loss": [], "val_loss": [], "train_iou": [], "val_iou": []}


def save_checkpoint(state, path):
    torch.save(state, path)
    print(f"    [Checkpoint] Saved -> {path}")


def load_checkpoint(path, model, optimizer=None):
    """Load weights and optionally the optimiser state for full resume."""
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    if optimizer and "optimizer_state" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state"])
    start = ckpt.get("epoch", 0) + 1
    hist  = ckpt.get("history", _empty_history())
    print(f"    [Checkpoint] Resumed from epoch {ckpt.get('epoch', 0) + 1}")
    return start, hist


def train(config):
    """
    Runs the full training pipeline for one model.

    Expects a config dict with keys like data_dir, model_type, epochs, lr, etc.
    Returns the trained model and the training history dict.
    """
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    scaler  = GradScaler("cuda") if use_amp else None
    print(f"[Train] Device : {device}  |  AMP : {use_amp}")

    train_loader, val_loader, _ = get_dataloaders(
        data_dir     = config["data_dir"],
        dataset_type = config.get("dataset_type", "generic"),
        img_size     = config.get("img_size", 128),
        batch_size   = config.get("batch_size", 16),
        num_workers  = config.get("num_workers", 0),
    )

    model = build_model(
        model_type   = config.get("model_type", "baseline"),
        img_size     = config.get("img_size", 128),
        dropout_rate = config.get("dropout_rate", 0.3),
    ).to(device)
    print(f"[Train] Model  : {config.get('model_type')}  |  params : {count_parameters(model):,}")

    loss_fn   = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=config.get("lr", 1e-3),
                            weight_decay=config.get("weight_decay", 1e-4))

    epochs        = config.get("epochs", 50)
    warmup_epochs = config.get("warmup_epochs", 3)
    scheduler     = _cosine_with_warmup(optimizer, warmup_epochs, epochs)
    clip_norm     = config.get("clip_norm", 1.0)

    ckpt_dir = Path(config.get("checkpoint_dir", "checkpoints"))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    run_name  = config.get("run_name", config.get("model_type", "model"))
    best_ckpt = ckpt_dir / f"{run_name}_best.pth"
    last_ckpt = ckpt_dir / f"{run_name}_last.pth"

    # Resume from last checkpoint if one exists and resume is not disabled
    start_epoch = 0
    history     = _empty_history()
    if config.get("resume", True) and last_ckpt.exists():
        start_epoch, history = load_checkpoint(last_ckpt, model, optimizer)
        # Fast-forward the LR scheduler to the correct epoch
        for _ in range(start_epoch):
            scheduler.step()

    patience     = config.get("patience", 10)
    best_val_iou = max(history["val_iou"]) if history["val_iou"] else 0.0
    no_improve   = 0

    print(f"\n[Train] Epochs={epochs}  Warmup={warmup_epochs}  Patience={patience}  ClipNorm={clip_norm}")
    print("=" * 72)

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        train_loss, train_iou = train_one_epoch(model, train_loader, optimizer, loss_fn, device, scaler, clip_norm)
        val_loss,   val_iou   = validate(model, val_loader, loss_fn, device)
        scheduler.step()
        lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_iou"].append(train_iou)
        history["val_iou"].append(val_iou)

        print(
            f"Epoch {epoch+1:03d}/{epochs}  "
            f"loss={train_loss:.4f}/{val_loss:.4f}  "
            f"iou={train_iou:.4f}/{val_iou:.4f}  "
            f"lr={lr:.2e}  ({time.time()-t0:.1f}s)"
        )

        # Always save the last checkpoint so we can resume if training is interrupted
        save_checkpoint(
            {"epoch": epoch, "model_state": model.state_dict(),
             "optimizer_state": optimizer.state_dict(),
             "history": history, "config": config},
            last_ckpt,
        )

        # Save the best checkpoint separately so evaluation always uses the best weights
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            no_improve   = 0
            save_checkpoint(
                {"epoch": epoch, "model_state": model.state_dict(),
                 "val_loss": val_loss, "val_iou": val_iou, "config": config},
                best_ckpt,
            )
            print(f"    [Best] val_iou={val_iou:.4f}")
        else:
            no_improve += 1
            print(f"    [EarlyStopping] {no_improve}/{patience}")
            if no_improve >= patience:
                print(f"\n[Train] Early stopping at epoch {epoch+1}.")
                break

    hist_path = ckpt_dir / f"{run_name}_history.json"
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"\n[Train] History -> {hist_path}")
    print(f"[Train] Best    -> {best_ckpt}  (val_iou={best_val_iou:.4f})")
    return model, history


def _parse_args():
    p = argparse.ArgumentParser(description="Train SOD model")
    p.add_argument("--data_dir",       required=True)
    p.add_argument("--dataset_type",   default="generic", choices=["generic", "duts", "ecssd", "msra10k"])
    p.add_argument("--model_type",     default="baseline", choices=["baseline", "improved"])
    p.add_argument("--img_size",       type=int,   default=128)
    p.add_argument("--batch_size",     type=int,   default=16)
    p.add_argument("--epochs",         type=int,   default=50)
    p.add_argument("--lr",             type=float, default=1e-3)
    p.add_argument("--weight_decay",   type=float, default=1e-4)
    p.add_argument("--patience",       type=int,   default=10)
    p.add_argument("--warmup_epochs",  type=int,   default=3)
    p.add_argument("--clip_norm",      type=float, default=1.0)
    p.add_argument("--dropout_rate",   type=float, default=0.3)
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--run_name",       default=None)
    p.add_argument("--no_resume",      action="store_true")
    p.add_argument("--num_workers",    type=int,   default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg  = {
        "data_dir":       args.data_dir,
        "dataset_type":   args.dataset_type,
        "model_type":     args.model_type,
        "img_size":       args.img_size,
        "batch_size":     args.batch_size,
        "epochs":         args.epochs,
        "lr":             args.lr,
        "weight_decay":   args.weight_decay,
        "patience":       args.patience,
        "warmup_epochs":  args.warmup_epochs,
        "clip_norm":      args.clip_norm,
        "dropout_rate":   args.dropout_rate,
        "checkpoint_dir": args.checkpoint_dir,
        "run_name":       args.run_name or args.model_type,
        "resume":         not args.no_resume,
        "num_workers":    args.num_workers,
    }
    train(cfg)
