"""
evaluate.py
Loads a trained model and measures how well it actually works.

We compute four standard metrics (IoU, Precision, Recall, F1) plus MAE,
then generate visual grids so you can see what the predictions look like
rather than just trusting numbers. There's also a comparison function that
puts two models side-by-side in a table and bar chart.
"""

import argparse
import json
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive so it works in scripts and Colab
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from data_loader import get_dataloaders
from sod_model import build_model


def compute_metrics(preds, targets, threshold=0.5, eps=1e-6):
    """
    Computes IoU, Precision, Recall, F1, and MAE for a batch.

    We binarise the predictions at `threshold` before computing overlap metrics,
    but keep the raw probabilities for MAE (which measures calibration).
    Returns a dict of scalar values averaged over the batch.
    """
    pred_bin = (preds > threshold).float()

    tp = (pred_bin * targets).sum(dim=(1, 2, 3))
    fp = (pred_bin * (1 - targets)).sum(dim=(1, 2, 3))
    fn = ((1 - pred_bin) * targets).sum(dim=(1, 2, 3))

    iou       = (tp + eps) / (tp + fp + fn + eps)
    precision = (tp + eps) / (tp + fp + eps)
    recall    = (tp + eps) / (tp + fn + eps)
    f1        = 2 * precision * recall / (precision + recall + eps)
    mae       = torch.abs(preds - targets).mean(dim=(1, 2, 3))

    return {
        "iou":       iou.mean().item(),
        "precision": precision.mean().item(),
        "recall":    recall.mean().item(),
        "f1":        f1.mean().item(),
        "mae":       mae.mean().item(),
    }


def evaluate_model(model, test_loader, device, threshold=0.5):
    """
    Runs the model over the entire test set and prints a summary table.
    Returns a dict with the aggregate metric scores.
    """
    model.eval()
    accum = {"iou": [], "precision": [], "recall": [], "f1": [], "mae": []}

    with torch.no_grad():
        for images, masks in tqdm(test_loader, desc="Evaluating", unit="batch"):
            images, masks = images.to(device), masks.to(device)
            preds = model(images)
            batch = compute_metrics(preds, masks, threshold)
            for k, v in batch.items():
                accum[k].append(v)

    results = {
        "IoU":       float(np.mean(accum["iou"])),
        "Precision": float(np.mean(accum["precision"])),
        "Recall":    float(np.mean(accum["recall"])),
        "F1":        float(np.mean(accum["f1"])),
        "MAE":       float(np.mean(accum["mae"])),
    }

    print("\n" + "=" * 42)
    print("  Test Set Results")
    print("=" * 42)
    for k, v in results.items():
        print(f"  {k:<12}: {v:.4f}")
    print("=" * 42)
    return results


def visualize_predictions(model, test_loader, device, num_samples=5,
                          save_dir="results", run_name="model", threshold=0.5):
    """
    Saves a grid of sample predictions showing:
      Input | Ground Truth | Predicted Saliency Map | Overlay

    This makes it easy to spot failure modes that metrics alone wouldn't reveal
    (e.g. the model might get the right IoU but predict a blurry blob).
    """
    model.eval()
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    images_list, masks_list, preds_list = [], [], []
    collected = 0

    with torch.no_grad():
        for images, masks in test_loader:
            preds = model(images.to(device)).cpu()
            images_list.append(images)
            masks_list.append(masks)
            preds_list.append(preds)
            collected += images.size(0)
            if collected >= num_samples:
                break

    images_all = torch.cat(images_list)[:num_samples]
    masks_all  = torch.cat(masks_list)[:num_samples]
    preds_all  = torch.cat(preds_list)[:num_samples]
    n = images_all.size(0)

    fig, axes = plt.subplots(n, 4, figsize=(16, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for col, title in enumerate(["Input Image", "Ground Truth", "Prediction", "Overlay"]):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold", pad=8)

    for i in range(n):
        img    = images_all[i].permute(1, 2, 0).numpy()
        gt     = masks_all[i, 0].numpy()
        pred   = preds_all[i, 0].numpy()
        pred_b = (pred > threshold).astype(np.float32)

        # Red-tinted overlay: highlight predicted salient region on the original image
        overlay = np.stack([
            np.clip(img[..., 0] + pred_b * 0.45, 0, 1),
            np.clip(img[..., 1] * (1 - pred_b * 0.35), 0, 1),
            np.clip(img[..., 2] * (1 - pred_b * 0.35), 0, 1),
        ], axis=-1)

        axes[i, 0].imshow(img)
        axes[i, 1].imshow(gt,      cmap="gray", vmin=0, vmax=1)
        axes[i, 2].imshow(pred,    cmap="hot",  vmin=0, vmax=1)
        axes[i, 3].imshow(overlay, vmin=0,      vmax=1)

        for j in range(4):
            axes[i, j].axis("off")

    plt.tight_layout()
    out_path = save_dir / f"{run_name}_predictions.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] Predictions saved -> {out_path}")


def plot_training_history(history, save_dir="results", run_name="model"):
    """Plots loss and IoU curves for both train and val sets across epochs."""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train")
    ax1.plot(epochs, history["val_loss"],   "r-o", markersize=4, label="Val")
    ax1.set_title("Loss", fontsize=13)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Combined Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, history["train_iou"], "b-o", markersize=4, label="Train")
    ax2.plot(epochs, history["val_iou"],   "r-o", markersize=4, label="Val")
    ax2.set_title("IoU", fontsize=13)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("IoU")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.suptitle(f"Training History - {run_name}", fontsize=14, y=1.01)
    plt.tight_layout()
    out_path = save_dir / f"{run_name}_training_curves.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] Training curves saved -> {out_path}")


def compare_models(results_dict, save_dir="results"):
    """
    Prints a side-by-side comparison table for all evaluated models
    and saves a bar chart. Pass a dict like {"baseline": metrics, "improved": metrics}.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    metrics = ["IoU", "Precision", "Recall", "F1", "MAE"]
    header  = f"{'Model':<22}" + "".join(f"{m:>12}" for m in metrics)
    sep     = "-" * len(header)

    print("\n" + "=" * len(sep))
    print("Model Comparison")
    print("=" * len(sep))
    print(header)
    print(sep)
    for name, res in results_dict.items():
        print(f"{name:<22}" + "".join(f"{res.get(m, 0.0):>12.4f}" for m in metrics))
    print("=" * len(sep))

    txt_path = save_dir / "comparison.txt"
    with open(txt_path, "w") as fh:
        fh.write(header + "\n" + sep + "\n")
        for name, res in results_dict.items():
            fh.write(f"{name:<22}" + "".join(f"{res.get(m, 0.0):>12.4f}" for m in metrics) + "\n")
    print(f"[Evaluate] Comparison table saved -> {txt_path}")

    # Bar chart — one subplot per metric
    n_models  = len(results_dict)
    colors    = plt.cm.Set2(np.linspace(0, 1, n_models))
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 5))

    for ax, metric in zip(axes, metrics):
        names  = list(results_dict.keys())
        values = [results_dict[n].get(metric, 0.0) for n in names]
        bars   = ax.bar(names, values, color=colors)
        ax.set_title(metric, fontsize=12)
        ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)
        ax.tick_params(axis="x", rotation=20)
        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.004,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=9)

    plt.suptitle("Model Comparison", fontsize=14, y=1.02)
    plt.tight_layout()
    chart_path = save_dir / "model_comparison.png"
    plt.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Evaluate] Comparison chart saved -> {chart_path}")


def run_full_evaluation(checkpoint_path, data_dir, dataset_type="generic",
                        img_size=128, batch_size=16, run_name="model",
                        save_dir="results", num_vis_samples=5, threshold=0.5):
    """
    One-stop function: loads a checkpoint, evaluates on test set,
    saves prediction visuals, and plots training curves if history exists.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt       = torch.load(checkpoint_path, map_location=device)
    cfg        = ckpt.get("config", {})
    model_type = cfg.get("model_type", "baseline")
    model_size = cfg.get("img_size", img_size)

    model = build_model(model_type=model_type, img_size=model_size)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    print(f"[Evaluate] Loaded {model_type} model from {checkpoint_path}")

    _, _, test_loader = get_dataloaders(data_dir=data_dir, dataset_type=dataset_type,
                                        img_size=img_size, batch_size=batch_size)

    metrics = evaluate_model(model, test_loader, device, threshold=threshold)
    visualize_predictions(model, test_loader, device, num_samples=num_vis_samples,
                          save_dir=save_dir, run_name=run_name, threshold=threshold)

    hist_path = Path(checkpoint_path).parent / f"{run_name}_history.json"
    if hist_path.exists():
        with open(hist_path) as fh:
            plot_training_history(json.load(fh), save_dir=save_dir, run_name=run_name)

    return metrics


def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate a trained SOD model")
    p.add_argument("--checkpoint",          required=True)
    p.add_argument("--data_dir",            required=True)
    p.add_argument("--dataset_type",        default="generic", choices=["generic", "duts", "ecssd", "msra10k"])
    p.add_argument("--img_size",            type=int,   default=128)
    p.add_argument("--batch_size",          type=int,   default=16)
    p.add_argument("--run_name",            default="model")
    p.add_argument("--save_dir",            default="results")
    p.add_argument("--num_vis",             type=int,   default=5)
    p.add_argument("--threshold",           type=float, default=0.5)
    p.add_argument("--compare_checkpoint",  default=None)
    p.add_argument("--compare_name",        default="improved")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    metrics_main = run_full_evaluation(
        checkpoint_path=args.checkpoint, data_dir=args.data_dir,
        dataset_type=args.dataset_type, img_size=args.img_size,
        batch_size=args.batch_size, run_name=args.run_name,
        save_dir=args.save_dir, num_vis_samples=args.num_vis, threshold=args.threshold,
    )

    if args.compare_checkpoint:
        metrics_cmp = run_full_evaluation(
            checkpoint_path=args.compare_checkpoint, data_dir=args.data_dir,
            dataset_type=args.dataset_type, img_size=args.img_size,
            batch_size=args.batch_size, run_name=args.compare_name,
            save_dir=args.save_dir, num_vis_samples=args.num_vis, threshold=args.threshold,
        )
        compare_models({args.run_name: metrics_main, args.compare_name: metrics_cmp},
                       save_dir=args.save_dir)
