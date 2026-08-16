"""Random-init control: token-axis vs clip-axis effective rank on the val split.

This is the control both adversarial reviewers named as decisive. If a randomly
initialised encoder already shows token rank ~381 and pooled rank ~11, then the
"33x axis discrepancy" measures the architecture, not the trained model.
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
ROOT = "."


def effective_rank(x: np.ndarray) -> float:
    x = x - x.mean(0, keepdims=True)
    cov = (x.T @ x) / max(1, len(x) - 1)
    w = np.linalg.eigvalsh(cov.astype(np.float64))
    w = np.clip(w, 0.0, None)
    total = w.sum()
    if total <= 0:
        return 0.0
    p = w / total
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def collect(encoder, loader, device, max_token_rows=200_000):
    pooled, tokens_pre, tokens_post = [], [], []
    rng = np.random.default_rng(0)
    with torch.inference_mode():
        for batch in loader:
            video = batch["video"] if isinstance(batch, dict) else batch[0]
            video = video.to(device)
            post, pre = encoder(video, return_pre_norm=True)
            pooled.append(pre.float().mean(dim=1).cpu().numpy())
            for store, tensor in ((tokens_pre, pre), (tokens_post, post)):
                flat = tensor.float().reshape(-1, tensor.size(-1)).cpu().numpy()
                keep = rng.choice(len(flat), size=min(2000, len(flat)), replace=False)
                store.append(flat[keep])
    return (
        np.concatenate(pooled),
        np.concatenate(tokens_pre)[:max_token_rows],
        np.concatenate(tokens_post)[:max_token_rows],
    )


def main():
    checkpoint_path = sys.argv[1]
    device = resolve_device("auto")
    checkpoint = load_checkpoint(checkpoint_path)
    config = checkpoint["config"]

    dataset = HealthGaitDataset(
        MANIFEST,
        split="val",
        root=ROOT,
        clip_length=int(config["num_frames"]),
        image_size=int(config["img_size"]),
        windows=3,
    )
    loader = DataLoader(dataset, batch_size=int(config["batch_size"]), num_workers=2)

    results = {}
    for name, encoder in (
        ("trained", build_target_encoder(checkpoint, device)),
        ("random_init", build_random_target_encoder(config, device, seed=0)),
    ):
        encoder.eval()
        pooled, tok_pre, tok_post = collect(encoder, loader, device)
        results[name] = {
            "clips": int(len(pooled)),
            "clip_pooled_pre_norm_rank": round(effective_rank(pooled), 2),
            "token_pre_norm_rank": round(effective_rank(tok_pre), 2),
            "token_post_norm_rank": round(effective_rank(tok_post), 2),
            "token_rows": int(len(tok_pre)),
        }
        print(name, json.dumps(results[name]), flush=True)

    print(json.dumps(results, indent=2))
    with open("notes/scripts/random_init_rank.json", "w") as handle:
        json.dump(results, handle, indent=2)


if __name__ == "__main__":
    main()
