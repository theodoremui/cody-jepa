from pathlib import Path
import csv
import random

root = Path("data/healthgait/raw/Health_Gait")
modality = "silhouette"
min_frames = 16
seed = 0
val_fraction = 0.2

rows = []

for trial_dir in sorted((root / modality).glob("PA*/**/*")):
    if not trial_dir.is_dir():
        continue

    frames = sorted(
        list(trial_dir.glob("*.jpg")) + 
        list(trial_dir.glob("*.png"))
    )

    if len(frames) < min_frames:
        continue

    rel = trial_dir.relative_to(root / modality)
    if len(rel.parts) < 3:
        continue

    subject_id = rel.parts[0]
    gait_system = rel.parts[1]
    trial = rel.parts[-1]

    rows.append({
        "subject_id": subject_id,
        "modality": modality,
        "gait_system": gait_system,
        "trial": trial,
        "frame_dir": str(trial_dir),
        "num_frames": len(frames),
    })

    subjects = sorted({row["subject_id"] for row in rows})
    rng = random.Random(seed)
    rng.shuffle(subjects)

    n_val = max(1, round(val_fraction * len(subjects)))
    val_subjects = set(subjects[:n_val])

    for row in rows:
        row["split"] = "val" if row["subject_id"] in val_subjects else "train"

    out = Path(f"data/healthgait/manifests/{modality}_subject_split_seed{seed}.csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "subject_id",
            "modality",
            "gait_system",
            "trial",
            "frame_dir",
            "num_frames",
            "split",
        ])
        writer.writeheader()
        writer.writerows(rows)

    train_subjects = {row["subject_id"] for row in rows if row["split"] == "train"}
    val_subjects = {row["subject_id"] for row in rows if row["split"] == "val"}

    print(f"wrote {len(rows)} clips to {out}")
    print(f"train subjects: {len(train_subjects)}")
    print(f"val subjects: {len(val_subjects)}")
    print(f"overlap: {train_subjects & val_subjects}")
