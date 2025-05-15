# %%
import math, os, glob, functools, random, time, json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torch.cuda.amp import autocast, GradScaler

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True  # sp<<eed

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on: {DEVICE}")

# data
TRAIN_DIR = "train"
VAL_DIR = "val"  # optional – can point to a split of train_meta
GEOM_CSV = "sensor_geometry.csv"

MAX_HITS = 2048  # down-sample hits to keep memory in check
BATCH_SIZE = 6
NUM_WORKERS = 4

# model
D_MODEL = 128
DEPTH = 6
N_HEADS = 8

# training
LR = 1e-4
EPOCHS = 150
WEIGHT_DECAY = 1e-2
GRAD_CLIP_NORM = 1.0

FEAT_DIM = 6  # ←  time, charge, aux, x, y, z

# %%_calss_IceCubeDataset


class IceCubeDataset(Dataset):
    """
    Streams events from batch_*.parquet → O(1) RAM.
    If is_train=False, targets tensor is dummy zeros.
    """

    def __init__(
        self, root_dir, geometry_csv, is_train=True, max_hits=None, use_aux=False
    ):
        self.files = sorted(Path(root_dir).glob("batch_*.parquet"))
        assert self.files, f"No parquet files found in {root_dir}"
        self.geometry = pd.read_csv(geometry_csv)[["x", "y", "z"]].to_numpy(np.float32)

        # read *_meta.parquet once (tiny)
        meta_path = Path(root_dir) / f"{Path(root_dir).name}_meta.parquet"
        self.meta = pq.read_table(meta_path).to_pandas() if is_train else None

        self.is_train = is_train
        self.max_hits = max_hits
        self.use_aux = use_aux

        # build index (file idx, row group idx)
        self._event_index = []
        for f_idx, f in enumerate(self.files):
            pf = pq.ParquetFile(f)
            self._event_index.extend([(f_idx, rg) for rg in range(pf.num_row_groups)])

    def __len__(self):
        return len(self._event_index)

    def __getitem__(self, idx):
        f_idx, rg = self._event_index[idx]
        table = pq.ParquetFile(self.files[f_idx]).read_row_group(rg)
        df = table.to_pandas()

        if not self.use_aux:
            df = df[~df["auxiliary"].values]

        xyz = self.geometry[df.sensor_id.values]
        feats = np.concatenate(
            [
                (df.time.values[:, None] - df.time.values.min()) * 1e-3,  # ns→µs
                df.charge.values[:, None],
                df.auxiliary.values[:, None].astype(np.float32),
                xyz,
            ],
            axis=1,
        ).astype(np.float32)

        # random down-sample for memory
        if self.max_hits and len(feats) > self.max_hits:
            keep = np.random.choice(len(feats), self.max_hits, replace=False)
            feats = feats[keep]

        feats = torch.from_numpy(feats)

        if self.is_train:
            meta = self.meta.iloc[idx]
            az, zen = meta.azimuth, meta.zenith
            target = torch.tensor(
                [
                    math.sin(zen) * math.cos(az),
                    math.sin(zen) * math.sin(az),
                    math.cos(zen),
                ],
                dtype=torch.float32,
            )
            return feats, target
        else:
            return feats, torch.zeros(3)


# %%_class_SequencePadCollate


class SequencePadCollate:
    """Pads variable-length sequences; returns x, mask, y."""

    def __init__(self, device=DEVICE):
        self.device = device

    def __call__(self, batch):
        xs, ys = zip(*batch)
        lens = torch.tensor([len(x) for x in xs]).to(self.device)
        xpad = pad_sequence(xs, batch_first=True).to(self.device)  # [B,L,7]
        mask = torch.arange(xpad.size(1), device=self.device)[None, :] >= lens[:, None]
        y = torch.stack(ys).to(self.device)
        return xpad, mask, y


# %%_class_HitEncoder


class HitEncoder(nn.Module):
    """Project 6-dim hit features to the model width."""

    def __init__(self, in_dim=FEAT_DIM, d_model=D_MODEL):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)  # single projection layer
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.proj(x))


# %%_class_IceFormer


class IceFormer(nn.Module):
    def __init__(self, d_model=D_MODEL, depth=DEPTH, nhead=N_HEADS):
        super().__init__()
        enc = nn.TransformerEncoderLayer(
            d_model,
            nhead,
            dim_feedforward=4 * d_model,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.ModuleList([enc for _ in range(depth)])

        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.out = nn.Linear(d_model, 3)

    def forward(self, x, mask):
        B = x.size(0)
        q = self.query.expand(B, -1, -1)  # prepend learnable token
        x = torch.cat([q, x], 1)
        mask = torch.cat([torch.zeros_like(mask[:, :1]), mask], 1)

        for blk in self.encoder:
            x = blk(x, src_key_padding_mask=mask)

        v = x[:, 0]  # query token
        v = v / v.norm(dim=-1, keepdim=True)  # unit
        return self.out(v)


def angular_loss(pred, tgt):
    cos = (pred * tgt).sum(-1).clamp(-1, 1)
    return torch.acos(cos).mean()


train_ds = IceCubeDataset(TRAIN_DIR, GEOM_CSV, is_train=True, max_hits=MAX_HITS)
val_ds = IceCubeDataset(VAL_DIR, GEOM_CSV, is_train=True, max_hits=MAX_HITS)

train_dl = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    collate_fn=SequencePadCollate(),
)
val_dl = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    collate_fn=SequencePadCollate(),
)


def run_epoch(model, loader, opt=None, scaler=None):
    is_train = opt is not None
    model.train() if is_train else model.eval()
    running = 0.0
    with torch.set_grad_enabled(is_train):
        for x, mask, y in loader:
            if is_train:
                opt.zero_grad(set_to_none=True)
            with autocast():
                pred = model(x, mask)
                loss = angular_loss(pred, y)
            if is_train:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
                scaler.step(opt)
                scaler.update()
            running += loss.item() * y.size(0)
    return running / len(loader.dataset)


def fit(model, train_dl, val_dl, epochs):
    opt = torch.optim.AdamW(model.parameters(), LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler()
    best = float("inf")

    for ep in range(1, epochs + 1):
        t0 = time.time()
        train_loss = run_epoch(model, train_dl, opt, scaler)
        val_loss = run_epoch(model, val_dl)

        if val_loss < best:
            best = val_loss
            torch.save(
                {
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "val_loss": best,
                },
                "iceformer_best.pth",
            )
            flag = "  ✅ saved"
        else:
            flag = ""
        dur = time.time() - t0
        print(
            f"Epoch {ep:02d}: train {train_loss:.4f} – val {val_loss:.4f}"
            f"  ({dur/60:.1f} min){flag}"
        )


# %%_class_IceCubeNet


class IceCubeNet(nn.Module):
    def __init__(self, d_model=D_MODEL, depth=DEPTH, nhead=N_HEADS):
        super().__init__()
        self.feat = HitEncoder(d_model=d_model)  # <- NEW definition just above
        self.backbone = IceFormer(d_model, depth, nhead)

    def forward(self, x, mask):
        x = self.feat(x)
        assert x.size(-1) == D_MODEL, "HitEncoder output size mismatch"
        return self.backbone(x, mask)


# debug: check shapes
dummy_x = torch.randn(2, 10, FEAT_DIM).to(DEVICE)  # (B=2, L=10, 6)
dummy_mask = torch.zeros(2, 10, dtype=torch.bool).to(DEVICE)

with torch.no_grad():
    out = model(dummy_x, dummy_mask)
print("Output shape:", out.shape)  #  → torch.Size([2, 3])


model = IceCubeNet().to(DEVICE)
fit(model, train_dl, val_dl, EPOCHS)


# %%_INFERENCE


def vec2angles(vec):
    x, y, z = vec.T
    az = np.arctan2(y, x) % (2 * np.pi)
    ze = np.arccos(z.clip(-1, 1))
    return az, ze


# Load best weights
ckpt = torch.load("./iceformer_best.pth", map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# TODO: build a test DataLoader just like val_dl but is_train=False
# test_dl = ...

pred_rows = []
with torch.no_grad():
    for x, mask, _ in test_dl:
        v = model(x, mask).cpu().numpy()
        az, ze = vec2angles(v)
        for eid, a, z in zip(event_ids_batch, az, ze):  # supply event_ids
            pred_rows.append((eid, a, z))

submission = pd.DataFrame(pred_rows, columns=["event_id", "azimuth", "zenith"])
submission.to_csv("submission.csv", index=False)
print("submission.csv written!")
