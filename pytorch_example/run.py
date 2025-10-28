#!/usr/bin/env python3
"""
Minimal PyTorch Lightning multi-GPU / multi-node demo.

Usage (single node, 8 GPUs):
    python run.py --num_nodes 1 --num_gpus 8
"""

import argparse
from typing import Optional, Any

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import lightning.pytorch as pl


# ----------------------------
# Synthetic dataset
# ----------------------------
class GaussianBlobs(Dataset):
    """
    Two-class blobs in 32-D. Easy to overfit/learn quickly.
    """
    def __init__(self, n_samples: int = 50_000, dim: int = 32, std: float = 1.0, seed: int = 123):
        g = torch.Generator().manual_seed(seed)
        self.dim = dim

        n0 = n_samples // 2
        n1 = n_samples - n0

        # Class 0 centered at -1, class 1 at +1
        self.x0 = torch.randn(n0, dim, generator=g) * std + (-1.0)
        self.x1 = torch.randn(n1, dim, generator=g) * std + (+1.0)

        self.y0 = torch.zeros(n0, dtype=torch.long)
        self.y1 = torch.ones(n1, dtype=torch.long)

        self.x = torch.cat([self.x0, self.x1], dim=0)
        self.y = torch.cat([self.y0, self.y1], dim=0)

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


# ----------------------------
# LightningModule
# ----------------------------
class TinyMLP(pl.LightningModule):
    def __init__(self, dim: int = 32, hidden: int = 128, num_layers: int = 2, lr: float = 3e-3):
        super().__init__()
        self.save_hyperparameters()
        layers = [
            nn.Linear(dim, hidden),
            nn.ReLU(),
        ]

        for _ in range(num_layers-1):
            layers.append(nn.Linear(hidden, hidden))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)
        self.net.add_module("final_layer", nn.Linear(hidden, 2))
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _step(self, batch: Any, stage: str):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._step(batch, "val")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)


# ----------------------------
# DataModule
# ----------------------------
class BlobsDataModule(pl.LightningDataModule):
    def __init__(self, dim=32, train_size=5_000_000, val_size=500_000, batch_size=1024, num_workers=4, seed=123):
        super().__init__()
        self.dim = dim
        self.train_size = train_size
        self.val_size = val_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seed = seed

        # Important so each process can re-generate deterministically
        self.generator = torch.Generator()

    def setup(self, stage: Optional[str] = None):
        self.train_ds = GaussianBlobs(n_samples=self.train_size, dim=self.dim, seed=self.seed)
        self.val_ds = GaussianBlobs(n_samples=self.val_size, dim=self.dim, seed=self.seed + 1)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )


# ----------------------------
# Main
# ----------------------------
def parse_args():
    p = argparse.ArgumentParser(description="PL multi-GPU / multi-node demo")
    p.add_argument("--num_nodes", type=int, required=True, help="Number of nodes for training")
    p.add_argument("--num_gpus", type=int, required=True, help="GPUs per node")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=1024)
    return p.parse_args()


def main():
    args = parse_args()

    pl.seed_everything(42, workers=True)

    dim = 4096
    datamodule = BlobsDataModule(batch_size=args.batch_size, dim=dim)
    model = TinyMLP(
        dim=dim,
        hidden=1024,
        num_layers=32,
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices=args.num_gpus,
        num_nodes=args.num_nodes,
        strategy='ddp',
        max_epochs=args.epochs,
        enable_progress_bar=True,
        benchmark=True,
    )

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    main()
