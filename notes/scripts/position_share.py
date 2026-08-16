"""How much of within-clip variance is the position main effect?

Every clip contributes the same fixed grid of token positions, so tokens form a
balanced two-way layout: x[i,p] = mu + a[i] + b[p] + e[i,p], where a[i] is the
clip effect, b[p] is the position effect, and e is the interaction and noise.

For a balanced layout the trace decomposition is exact:

    tr(Sigma_token) = tr(Sigma_clip) + tr(Sigma_position) + tr(Sigma_residual)

and Sigma_within = Sigma_position + Sigma_residual. So the position share of
within-clip variance is tr(Sigma_position) / tr(Sigma_within).

This tests the mechanism hypothesis directly on an existing checkpoint, before
any new training run. If position explains most of the within-clip variance, the
hypothesis has preliminary support. If it does not, the hypothesis must be
demoted and said so plainly.

One pass over the data. Memory is O(N*d + T*d), not O(N*T*d).
"""

import json
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, "/Users/theodoremui/dev/cody-jepa")

from cody_jepa.data import HealthGaitDataset
from cody_jepa.engine import load_checkpoint, resolve_device
from cody_jepa.evaluation import build_random_target_encoder, build_target_encoder

MANIFEST = "data/healthgait/manifests/silhouette_gfc_candidate_seed0.csv"
OUT = "notes/scripts/position_share.json"


def accumulate(encoder, loader, device):
    clip_sums, pos_sum, total_sq, n_clips, n_pos, dim = [], None, 0.0, 0, None, None
    with torch.inference_mode():
        for batch in loader:
            video = (batch["video"] if isinstance(batch, dict) else batch[0]).to(device)
            _, pre = encoder(video, return_pre_norm=True)
            x = pre.float().cpu().double()                 # (B, T, d), MPS has no float64
            if n_pos is None:
                n_pos, dim = x.shape[1], x.shape[2]
            clip_sums.append(x.sum(dim=1).numpy())         # (B, d)
            step = x.sum(dim=0).numpy()                    # (T, d)
            pos_sum = step if pos_sum is None else pos_sum + step
            total_sq += float((x * x).sum())
            n_clips += x.shape[0]
    return np.concatenate(clip_sums), pos_sum, total_sq, n_clips, n_pos, dim


def decompose(encoder, loader, device):
    C, P, Q, N, T, d = accumulate(encoder, loader, device)
    mu = C.sum(axis=0) / (N * T)
    mu_sq = float(mu @ mu)

    tr_total = Q / (N * T) - mu_sq
    clip_means = C / T
    tr_clip = float((clip_means * clip_means).sum()) / N - mu_sq
    pos_means = P / N
    tr_pos = float((pos_means * pos_means).sum()) / T - mu_sq
    tr_within = tr_total - tr_clip
    tr_resid = tr_within - tr_pos

    return {
        "clips": int(N),
        "positions_per_clip": int(T),
        "dim": int(d),
        "trace_total": round(tr_total, 5),
        "trace_clip_between": round(tr_clip, 5),
        "trace_within": round(tr_within, 5),
        "trace_position_effect": round(tr_pos, 5),
        "trace_residual": round(tr_resid, 5),
        "beta_between_share": round(tr_clip / tr_total, 6),
        "position_share_of_within": round(tr_pos / tr_within, 5),
        "residual_share_of_within": round(tr_resid / tr_within, 5),
    }


def main():
    device = resolve_device("auto")
    out = {}
    pairs = list(zip(sys.argv[1::2], sys.argv[2::2]))
    for label, path in pairs:
        ck = load_checkpoint(path)
        cfg = ck["config"]
        ds = HealthGaitDataset(
            MANIFEST, split="val", root=".",
            clip_length=int(cfg["num_frames"]), image_size=int(cfg["img_size"]), windows=3,
        )
        loader = DataLoader(ds, batch_size=int(cfg["batch_size"]), num_workers=2)
        out[label] = decompose(build_target_encoder(ck, device).eval(), loader, device)
        print(label, json.dumps(out[label]), flush=True)
        if label == "a00_no_reg":
            rnd = build_random_target_encoder(cfg, device, seed=0).eval()
            out["random_init"] = decompose(rnd, loader, device)
            print("random_init", json.dumps(out["random_init"]), flush=True)
    with open(OUT, "w") as handle:
        json.dump(out, handle, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
