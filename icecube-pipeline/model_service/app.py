#%%
from flask import Flask, request, jsonify
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import os

# --- Model definitions (replicate your trained model) ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters (must match training)
D_MODEL = 128
DEPTH = 6
N_HEADS = 8
FEAT_DIM = 6  # time, charge, auxiliary, x, y, z

class HitEncoder(nn.Module):
    def __init__(self, in_dim=FEAT_DIM, d_model=D_MODEL):
        super().__init__()
        self.proj = nn.Linear(in_dim, d_model)
        self.act = nn.SiLU()
    def forward(self, x):
        return self.act(self.proj(x))

class IceFormer(nn.Module):
    def __init__(self, d_model=D_MODEL, depth=DEPTH, nhead=N_HEADS):
        super().__init__()
        enc = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=4*d_model,
            batch_first=True, norm_first=True
        )
        self.encoder = nn.ModuleList([enc for _ in range(depth)])
        self.query = nn.Parameter(torch.randn(1, 1, d_model))
        self.out = nn.Linear(d_model, 3)
    def forward(self, x, mask):
        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        x = torch.cat([q, x], dim=1)
        mask = torch.cat([torch.zeros(B,1, dtype=torch.bool, device=x.device), mask], dim=1)
        for blk in self.encoder:
            x = blk(x, src_key_padding_mask=mask)
        v = x[:, 0]
        v = v / v.norm(dim=-1, keepdim=True)
        return self.out(v)

class IceCubeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.feat = HitEncoder()
        self.backbone = IceFormer()
    def forward(self, x, mask):
        x = self.feat(x)
        return self.backbone(x, mask)

class SequencePadCollate:
    """Pads variable-length sequences; returns xpad, mask, dummy y."""
    def __init__(self, device=DEVICE):
        self.device = device
    def __call__(self, batch):
        xs, ys = zip(*batch)
        lengths = [x.size(0) for x in xs]
        xpad = pad_sequence(xs, batch_first=True).to(self.device)
        mask = torch.arange(xpad.size(1), device=self.device)[None, :] >= torch.tensor(lengths, device=self.device)[:, None]
        y = torch.stack(ys).to(self.device)
        return xpad, mask, y

# --- Load geometry and model ---
sensor_geometry = pd.read_csv("sensor_geometry.csv")[['x','y','z']].to_numpy(np.float32)
model = IceCubeNet().to(DEVICE)
ckpt = torch.load("iceformer_best.pth", map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
collate_fn = SequencePadCollate(device=DEVICE)

# --- Utility to convert vector output to angles ---
def vec2angles(vec: np.ndarray):
    x, y, z = vec.T
    az = np.arctan2(y, x) % (2 * np.pi)
    ze = np.arccos(np.clip(z, -1, 1))
    return az, ze

# --- Flask application ---
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    payload = request.get_json(force=True)
    event_id = payload.get('event_id')
    records = payload.get('data', [])

    # Build dataframe and feature tensor
    df = pd.DataFrame(records)
    # normalize time to microseconds
    times = (df['time'].values - df['time'].min()) * 1e-3
    charges = df['charge'].values
    aux = df['auxiliary'].astype(np.float32).values
    xyz = sensor_geometry[df['sensor_id'].values]
    feats = np.stack([times, charges, aux], axis=1)
    feats = np.concatenate([feats, xyz], axis=1).astype(np.float32)
    x = torch.from_numpy(feats).to(DEVICE)

    # pad and mask
    xpad, mask, _ = collate_fn([(x, torch.zeros(3))])

    # inference
    with torch.no_grad():
        out = model(xpad, mask).cpu().numpy()
    az, ze = vec2angles(out)

    # prepare response
    response = {
        'event_id': event_id,
        'azimuth': float(az[0]),
        'zenith': float(ze[0])
    }
    return jsonify(response)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)