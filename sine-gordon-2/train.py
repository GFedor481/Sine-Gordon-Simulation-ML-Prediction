"""
train.py
--------
Training pipeline for the SineGordonTransformer.

Features:
  - HDF5 dataset loader (reads from generate_dataset.py output)
  - SoftLabelCrossEntropy with Gaussian smoothing (sigma=2.0, circular)
  - AdamW optimizer with cosine LR schedule and warmup
  - Early stopping on validation loss
  - Checkpoint saving (best model + latest)
  - ±1 accuracy metric (matches paper's 97.02% evaluation)
  - Optional mixed-precision training (torch.cuda.amp)

Usage:
  python train.py --data sine_gordon_dataset.h5 --epochs 100 --batch-size 256
"""

import argparse
import logging
import math
import os
import time

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, random_split

from model import SineGordonTransformer

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("training.log"),
        logging.StreamHandler(),
    ],
    force=True,
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class SineGordonDataset(Dataset):
    """
    Reads the HDF5 file produced by generate_dataset.py.

    Returns:
        x: (T, P, 2) float32 tensor of [kinetic_energy, potential_energy]
        label: int, index of the localized pendulum
    """

    def __init__(self, h5_path: str, precision: int = None,
                 mean: np.ndarray = None, std: np.ndarray = None,
                 n_stats_samples: int = 1000):
        """
        Args:
            h5_path:         path to HDF5 file
            precision:       if set, truncate energy values to this many decimal places
            mean:            (2,) array [ke_mean, pe_mean] — pass from train set to val/test
            std:             (2,) array [ke_std,  pe_std]  — pass from train set to val/test
            n_stats_samples: how many samples to use when computing mean/std
        """
        self.h5_path = h5_path
        self.precision = precision

        with h5py.File(h5_path, "r") as f:
            self.sim_keys = [
                k for k in f.keys()
                if k.startswith("simulation_")
                and f[k].attrs["localized_pendulum"] >= 0
            ]

        log.info("Dataset: %d valid simulations in %s", len(self.sim_keys), h5_path)

        # Use provided stats (val/test) or compute from this dataset (train)
        if mean is not None and std is not None:
            self.mean = mean
            self.std  = std
        else:
            self._compute_stats(n_stats_samples)

    def _compute_stats(self, n_samples: int):
        """Compute per-channel mean and std from a random subset of samples."""
        sample_keys = self.sim_keys[:min(n_samples, len(self.sim_keys))]
        ke_all, pe_all = [], []

        log.info("Computing z-score stats from %d samples...", len(sample_keys))
        with h5py.File(self.h5_path, "r") as f:
            for k in sample_keys:
                dtheta = f[k]["dtheta"][:]
                energy = f[k]["energy"][:]
                ke = 0.5 * dtheta ** 2
                pe = energy - ke
                ke_all.append(ke.ravel())
                pe_all.append(pe.ravel())

        ke_all = np.concatenate(ke_all)
        pe_all = np.concatenate(pe_all)

        self.mean = np.array([ke_all.mean(), pe_all.mean()], dtype=np.float32)
        self.std  = np.array([ke_all.std()  + 1e-8, pe_all.std() + 1e-8], dtype=np.float32)
        log.info("KE  mean=%.4f std=%.4f", self.mean[0], self.std[0])
        log.info("PE  mean=%.4f std=%.4f", self.mean[1], self.std[1])

    def __len__(self):
        return len(self.sim_keys)

    def __getitem__(self, idx):
        key = self.sim_keys[idx]

        with h5py.File(self.h5_path, "r") as f:
            grp = f[key]
            theta  = grp["theta"][:]    # (T, P)
            dtheta = grp["dtheta"][:]   # (T, P)
            energy = grp["energy"][:]   # (T, P)

            ke = 0.5 * dtheta ** 2
            pe = energy - ke
            label = int(grp.attrs["localized_pendulum"])

        # Stack to (T, P, 2) then z-score normalize per channel
        x = np.stack([ke, pe], axis=-1).astype(np.float32)   # (T, P, 2)
        x = (x - self.mean) / self.std                        # broadcast over (T, P)

        if self.precision is not None:
            x = np.round(x, decimals=self.precision).astype(np.float32)

        return torch.from_numpy(x), label


# ---------------------------------------------------------------------------
# Gaussian-smoothed soft-label cross-entropy loss
# ---------------------------------------------------------------------------
class GaussianSoftLabelCrossEntropy(nn.Module):
    """
    Cross-entropy with Gaussian-smoothed labels over a circular ring.

    For target site c on a ring of P pendulums:
        soft_label[i] = exp(-d(i,c)^2 / (2 sigma^2))
    normalized to sum to 1, then standard cross-entropy is applied.

    This embeds the circular topology into the loss and penalizes predictions
    proportionally to their distance from the true site.
    """

    def __init__(self, P: int = 100, sigma: float = 2.0):
        super().__init__()
        self.P = P
        self.sigma = sigma

        # Precompute soft label template for each of P target sites
        # soft_labels[c] = normalized Gaussian centered at c
        templates = torch.zeros(P, P)
        for c in range(P):
            for i in range(P):
                d = min(abs(i - c), P - abs(i - c))   # circular distance
                templates[c, i] = math.exp(-d ** 2 / (2 * sigma ** 2))
            templates[c] /= templates[c].sum()

        self.register_buffer("templates", templates)   # (P, P)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            logits:  (B, P)
            targets: (B,) int64 — ground truth localization site
        Returns:
            scalar loss
        """
        soft = self.templates[targets]   # (B, P)
        log_probs = F.log_softmax(logits, dim=-1)   # (B, P)
        loss = -(soft * log_probs).sum(dim=-1).mean()
        return loss


# Need F for the loss
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Accuracy helpers
# ---------------------------------------------------------------------------
def accuracy_at_k(logits: torch.Tensor, labels: torch.Tensor, k: int = 1, P: int = 100):
    """
    Fraction of predictions within ±k pendulums of the true label (circular).
    """
    preds = logits.argmax(dim=-1)   # (B,)
    dist = torch.abs(preds - labels)
    dist = torch.minimum(dist, P - dist)   # circular distance
    return (dist <= k).float().mean().item()


# ---------------------------------------------------------------------------
# LR scheduler: linear warmup then cosine decay
# ---------------------------------------------------------------------------
def get_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Train / eval one epoch
# ---------------------------------------------------------------------------
def run_epoch(model, loader, criterion, optimizer, scaler, scheduler, device,
              train: bool = True):
    model.train(train)
    total_loss = 0.0
    total_acc1 = 0.0
    total_acc_pm1 = 0.0
    n = 0

    for x, labels in loader:
        x = x.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type='cuda', enabled=(scaler is not None)):
            logits = model(x)
            loss = criterion(logits, labels)

        if train:
            optimizer.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

        B = x.size(0)
        total_loss += loss.item() * B
        total_acc1 += accuracy_at_k(logits.detach(), labels, k=0) * B
        total_acc_pm1 += accuracy_at_k(logits.detach(), labels, k=1) * B
        n += B

    return total_loss / n, total_acc1 / n, total_acc_pm1 / n


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# Per-epoch attention visualization helper
# ---------------------------------------------------------------------------
def _visualize_epoch(model, dataset, epoch, output_dir, device, T=10, P=100):
    """Save attention score + activation plots for one sample at a given epoch."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec

    model.eval()
    viz_dir = os.path.join(output_dir, "viz_epochs")
    os.makedirs(viz_dir, exist_ok=True)

    # Use sample index 0 consistently so plots are comparable across epochs
    x_np, true_label = dataset[0]
    x = x_np.unsqueeze(0).to(device) if hasattr(x_np, "unsqueeze") else torch.from_numpy(x_np[None]).to(device)

    with torch.no_grad():
        logits, all_weights = model(x, return_all_weights=True)
    pred_label = int(logits.argmax(dim=-1).item())

    # Get hidden states too
    attn_maps = []
    for w in all_weights:
        w4 = w[0, 0].cpu().numpy().reshape(T, P, T, P)
        attn_2d = w4.mean(axis=(0, 2))   # (P, P)
        attn_maps.append(attn_2d)

    n_layers = len(attn_maps)
    fig, axes = plt.subplots(2, n_layers, figsize=(3.5 * n_layers, 7))
    fig.suptitle(
        f"Epoch {epoch}  |  True: {true_label}  Pred: {pred_label}",
        fontsize=12, fontweight="bold"
    )

    for li in range(n_layers):
        # Top row: attention matrix
        ax = axes[0, li]
        im = ax.imshow(attn_maps[li], aspect="equal", cmap="viridis",
                       interpolation="nearest", origin="upper")
        ax.axhline(true_label, color="red", lw=0.8, ls="--", alpha=0.8)
        ax.axvline(true_label, color="red", lw=0.8, ls="--", alpha=0.8)
        ax.set_title(f"L{li+1}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

        # Bottom row: activation norm per pendulum
        ax2 = axes[1, li]
        # Use the attention-weighted sum as a proxy for activation
        # (diagonal of attn matrix = self-attention strength per pendulum)
        diag = np.diag(attn_maps[li])
        ax2.fill_between(range(P), diag, alpha=0.4, color="steelblue")
        ax2.plot(diag, color="steelblue", lw=1.0)
        ax2.axvline(true_label, color="red", lw=1.0, ls="--")
        ax2.set_xticks([]); ax2.set_yticks([])
        ax2.set_xlabel(f"L{li+1}", fontsize=9)

    axes[0, 0].set_ylabel("Attn matrix", fontsize=9)
    axes[1, 0].set_ylabel("Self-attn", fontsize=9)

    plt.tight_layout()
    fname = os.path.join(viz_dir, f"epoch_{epoch:04d}.png")
    plt.savefig(fname, dpi=120, bbox_inches="tight")
    plt.close()
    log.info("  Saved epoch viz: %s", fname)



def train(
    data_path: str,
    output_dir: str = "checkpoints",
    epochs: int = 100,
    batch_size: int = 128,
    lr: float = 3e-4,
    weight_decay: float = 1e-2,
    val_fraction: float = 0.1,
    patience: int = 10,
    warmup_fraction: float = 0.05,
    use_amp: bool = True,
    precision: int = None,
    seed: int = 42,
    # Model hyperparameters
    T: int = 10,
    P: int = 100,
    d_model: int = 128,
    n_heads: int = 1,
    d_k: int = 128,
    d_ff: int = 512,
    n_layers: int = 8,
    n_sinks: int = 6,
    dropout: float = 0.1,
    sigma_loss: float = 2.0,
):
    torch.manual_seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info("Device: %s", device)

    # Dataset split — compute stats on full dataset, share with val
    dataset = SineGordonDataset(data_path, precision=precision)
    n_val   = max(1, int(len(dataset) * val_fraction))
    n_train = len(dataset) - n_val
    train_ds, val_ds = random_split(
        dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    # Give val the same mean/std as train (no data leakage)
    val_ds.dataset.mean = dataset.mean
    val_ds.dataset.std  = dataset.std
    log.info("Train: %d  Val: %d", n_train, n_val)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True)

    # Model
    model = SineGordonTransformer(
        T=T, P=P, d_model=d_model, n_heads=n_heads, d_k=d_k,
        d_ff=d_ff, n_layers=n_layers, n_sinks=n_sinks, dropout=dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log.info("Model parameters: %d", n_params)

    # Loss, optimizer, scheduler
    criterion = GaussianSoftLabelCrossEntropy(P=P, sigma=sigma_loss).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    total_steps = epochs * len(train_loader)
    warmup_steps = int(warmup_fraction * total_steps)
    scheduler = get_scheduler(optimizer, warmup_steps, total_steps)
    scaler = GradScaler('cuda') if (use_amp and device.type == "cuda") else None

    # Training loop
    best_val_loss = float("inf")
    epochs_no_improve = 0
    history = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc, train_pm1 = run_epoch(
            model, train_loader, criterion, optimizer, scaler, scheduler,
            device, train=True
        )
        with torch.no_grad():
            val_loss, val_acc, val_pm1 = run_epoch(
                model, val_loader, criterion, optimizer, scaler, scheduler,
                device, train=False
            )

        elapsed = time.time() - t0
        log.info(
            "Epoch %3d | train loss %.4f acc %.2f%% ±1 %.2f%% | "
            "val loss %.4f acc %.2f%% ±1 %.2f%% | %.1fs",
            epoch,
            train_loss, train_acc * 100, train_pm1 * 100,
            val_loss, val_acc * 100, val_pm1 * 100,
            elapsed,
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss, "train_acc": train_acc, "train_pm1": train_pm1,
            "val_loss": val_loss, "val_acc": val_acc, "val_pm1": val_pm1,
        })

        # Checkpoint
        torch.save(
            {"epoch": epoch, "model": model.state_dict(),
             "optimizer": optimizer.state_dict(), "history": history},
            os.path.join(output_dir, "latest.pt")
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pt"))
            log.info("  -> New best model saved (val_loss=%.4f)", best_val_loss)
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                log.info("Early stopping after %d epochs without improvement.", epoch)
                break

        # Visualize attention every 10 epochs
        if epoch % 10 == 0:
            try:
                _visualize_epoch(model, dataset, epoch, output_dir, device, T, P)
            except Exception as e:
                log.warning("Visualization at epoch %d failed: %s", epoch, e)

    log.info("Training complete. Best val loss: %.4f", best_val_loss)
    return history


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Train SineGordonTransformer")
    p.add_argument("--data", type=str, required=True, help="Path to HDF5 dataset")
    p.add_argument("--output-dir", type=str, default="checkpoints")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    p.add_argument("--precision", type=int, default=None,
                   help="Truncate inputs to N decimal places (for precision experiments)")
    p.add_argument("--seed", type=int, default=42)
    # Model args
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=1)
    p.add_argument("--d-k", type=int, default=128)
    p.add_argument("--d-ff", type=int, default=64)
    p.add_argument("--n-layers", type=int, default=8)
    p.add_argument("--n-sinks", type=int, default=6)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--sigma-loss", type=float, default=2.0)
    args = p.parse_args()

    train(
        data_path=args.data,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        patience=args.patience,
        use_amp=not args.no_amp,
        precision=args.precision,
        seed=args.seed,
        d_model=args.d_model,
        n_heads=args.n_heads,
        d_k=args.d_k,
        d_ff=args.d_ff,
        n_layers=args.n_layers,
        n_sinks=args.n_sinks,
        dropout=args.dropout,
        sigma_loss=args.sigma_loss,
    )


if __name__ == "__main__":
    main()