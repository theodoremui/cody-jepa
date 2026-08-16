"""Decompose token-axis covariance into within-clip and between-clip parts.

Law of total covariance for tokens x_{i,t} with equal tokens per clip:
    Sigma_token = Sigma_between + Sigma_within
where Sigma_between is the covariance of the clip means, which is exactly the
object a mean-pooled probe consumes. So the token-axis health metric is an
entropy over the eigenvalues of a SUM, and the between-clip share tells you how
much of it is measuring anything a probe can ever see.
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


def erank(cov):
    w = np.linalg.eigvalsh(cov.astype(np.float64))
    w = np.clip(w, 0.0, None)
    s = w.sum()
    if s <= 0:
        return 0.0
    p = w / s
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def analyse(encoder, loader, device):
    clip_means, within_sum, n_clips, n_tok = [], None, 0, 0
    with torch.inference_mode():
        for batch in loader:
            video = (batch["video"] if isinstance(batch, dict) else batch[0]).to(device)
            _, pre = encoder(video, return_pre_norm=True)
            x = pre.float().cpu().numpy().astype(np.float64)      # (B, T, D)
            c = x.mean(axis=1)                                     # (B, D)
            clip_means.append(c)
            dev = (x - c[:, None, :]).reshape(-1, x.shape[-1])     # centred within clip
            within_sum = dev.T @ dev if within_sum is None else within_sum + dev.T @ dev
            n_clips += x.shape[0]
            n_tok += dev.shape[0]

    C = np.concatenate(clip_means)
    Sb = np.cov(C, rowvar=False, bias=True)                        # between-clip
    Sw = within_sum / n_tok                                        # within-clip
    St = Sb + Sw                                                   # total token covariance

    tb, tw, tt = np.trace(Sb), np.trace(Sw), np.trace(St)
    return {
        "clips": int(n_clips),
        "tokens": int(n_tok),
        "trace_between": round(float(tb), 4),
        "trace_within": round(float(tw), 4),
        "between_share_of_token_variance": round(float(tb / tt), 5),
        "erank_token_total": round(erank(St), 2),
        "erank_between_pooled": round(erank(Sb), 2),
        "erank_within": round(erank(Sw), 2),
    }


def main():
    device = resolve_device("auto")
    out = {}
    for label, path in [(a, b) for a, b in zip(sys.argv[1::2], sys.argv[2::2])]:
        ck = load_checkpoint(path)
        cfg = ck["config"]
        ds = HealthGaitDataset(
            MANIFEST, split="val", root=".",
            clip_length=int(cfg["num_frames"]), image_size=int(cfg["img_size"]), windows=3,
        )
        loader = DataLoader(ds, batch_size=int(cfg["batch_size"]), num_workers=2)
        enc = build_target_encoder(ck, device).eval()
        out[label] = analyse(enc, loader, device)
        print(label, json.dumps(out[label]), flush=True)
        if label == "b02_clip_var":
            rnd = build_random_target_encoder(cfg, device, seed=0).eval()
            out["random_init"] = analyse(rnd, loader, device)
            print("random_init", json.dumps(out["random_init"]), flush=True)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
