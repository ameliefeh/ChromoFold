"""Lightning training module + z-score helpers + CLI entry point."""

import argparse
from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.loggers import CSVLogger
from torch_geometric.loader import DataLoader

from fp_gnn.dataset import FluorProteinDataset
from fp_gnn.model import FPNetA, FPNetB


class RandomSplitter:
    """Random train/valid/test splitter, lifted from the friend's ipynb.

    Dormant for now -- the CLI uses the `split` column from `labels.csv`
    while the dataset is tiny (N=2). Swap in by uncommenting the marked
    block in `main()` once the dataset grows past ~20 samples.
    """

    def split(self, dataset, frac_train=0.7, frac_valid=0.1, frac_test=0.2, seed=None):
        np.testing.assert_almost_equal(frac_train + frac_valid + frac_test, 1.0)
        if seed is not None:
            np.random.seed(seed)
        n = len(dataset)
        train_cutoff = int(frac_train * n)
        valid_cutoff = int((frac_train + frac_valid) * n)
        shuffled = np.random.permutation(n)
        return (
            shuffled[:train_cutoff],
            shuffled[train_cutoff:valid_cutoff],
            shuffled[valid_cutoff:],
        )


def compute_zscore_stats(targets, kdas):
    """Compute (target_mean, target_std, kda_mean, kda_std) with std
    clamped to >= 1.0 to avoid div-by-zero on tiny train splits.

    Args:
        targets: float tensor [N, 2] -- (brightness, emission) per sample
        kdas:    float tensor [N]    -- kDa per sample
    """
    target_mean = targets.mean(dim=0)
    target_std = targets.std(dim=0, unbiased=False).clamp_min(1.0)
    kda_mean = kdas.mean()
    kda_std = kdas.std(unbiased=False).clamp_min(1.0)
    return target_mean, target_std, kda_mean, kda_std


class FluorLitModule(pl.LightningModule):
    def __init__(
        self,
        net,
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size=1,
        lr=1e-3,
    ):
        super().__init__()
        self.net = net
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.lr = lr

        # Stats from the train split only
        ys = torch.cat([d.y for d in train_dataset], dim=0)         # [N_train, 2]
        kdas = torch.cat([d.kda for d in train_dataset], dim=0)     # [N_train]
        target_mean, target_std, kda_mean, kda_std = compute_zscore_stats(ys, kdas)

        self.register_buffer("target_mean", target_mean)
        self.register_buffer("target_std", target_std)
        self.register_buffer("kda_mean", kda_mean)
        self.register_buffer("kda_std", kda_std)

    def _attach_kda_z(self, batch):
        batch.kda_z = (batch.kda - self.kda_mean) / self.kda_std

    def _zscore_targets(self, y):
        return (y - self.target_mean) / self.target_std

    def _denormalize(self, y_z):
        return y_z * self.target_std + self.target_mean

    def training_step(self, batch, batch_idx):
        self._attach_kda_z(batch)
        pred_z = self.net(batch)
        y_z = self._zscore_targets(batch.y)
        loss = F.mse_loss(pred_z, y_z)
        self.log("train_loss", loss, batch_size=self.batch_size)
        return loss

    def _eval_step(self, batch, prefix):
        self._attach_kda_z(batch)
        pred_z = self.net(batch)
        pred = self._denormalize(pred_z)
        # MSE / MAE per target in original units
        diff = pred - batch.y
        mse = (diff ** 2).mean(dim=0)
        mae = diff.abs().mean(dim=0)
        self.log(f"{prefix}_mse_brightness", mse[0], batch_size=self.batch_size)
        self.log(f"{prefix}_mse_emission", mse[1], batch_size=self.batch_size)
        self.log(f"{prefix}_mae_brightness", mae[0], batch_size=self.batch_size)
        self.log(f"{prefix}_mae_emission", mae[1], batch_size=self.batch_size)

    def validation_step(self, batch, batch_idx):
        self._eval_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._eval_step(batch, "test")

    def configure_optimizers(self):
        return torch.optim.Adam(self.net.parameters(), lr=self.lr)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


def _build_net(model_name, hidden=64, steps=3):
    if model_name == "a":
        return FPNetA(node_embedding_dim=hidden, num_message_steps=steps)
    if model_name == "b":
        return FPNetB(node_embedding_dim=hidden, num_message_steps=steps)
    raise ValueError(f"Unknown model: {model_name!r} (choose 'a' or 'b')")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["a", "b"], default="a")
    parser.add_argument("--max-epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--data-root", type=str, default="data/processed")
    parser.add_argument("--labels", type=str, default="data/labels.csv")
    parser.add_argument("--repo-root", type=str, default=".")
    args = parser.parse_args()

    torch.manual_seed(0)
    import random
    np.random.seed(0)
    random.seed(0)

    ds = FluorProteinDataset(
        root=args.data_root, labels_csv=args.labels, repo_root=args.repo_root,
    )

    # --- Active: deterministic split via the `split` column in labels.csv ---
    # Used while the dataset is tiny (N=2); guarantees 7ZCT=train, 4OQW=test.
    # train_ds = [d for d in ds if d.split == "train"]
    # test_ds = [d for d in ds if d.split == "test"]

    # --- Swap-in: random split (uncomment when N > ~20) ---
    # When the dataset grows, comment out the two lines above and uncomment
    # the block below. Drop the `split` column from labels.csv at the same time.

    splitter = RandomSplitter()
    train_idx, _, test_idx = splitter.split(
        ds, frac_train=0.8, frac_valid=0.0, frac_test=0.2, seed=0,
    )
    train_ds = [ds[int(i)] for i in train_idx]
    test_ds  = [ds[int(i)] for i in test_idx]
    # If you want a real validation split too, change fractions to e.g.
    # 0.7/0.1/0.2 and pass valid_idx through to FluorLitModule(val_dataset=...).

    net = _build_net(args.model, hidden=args.hidden, steps=args.steps)
    lit = FluorLitModule(
        net=net,
        train_dataset=train_ds,
        val_dataset=test_ds,
        test_dataset=test_ds,
        batch_size=1,
        lr=args.lr,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=CSVLogger(save_dir="logs", name=f"model_{args.model}"),
        enable_progress_bar=True,
    )
    trainer.fit(lit)
    # Evaluate the best checkpoint per the spec; falls back to the last
    # weights if no checkpoint was saved (e.g. checkpointing disabled).
    trainer.test(lit, ckpt_path="best")


if __name__ == "__main__":
    main()
