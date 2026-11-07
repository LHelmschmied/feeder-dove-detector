from pathlib import Path
import pandas as pd

ROOT = Path("data/raw_clips")
FOLDER_TO_LABEL = {"dove_clips": "dove", "not_dove_clips": "other"}

rows = []
for folder, label in FOLDER_TO_LABEL.items():
    for clip in sorted((ROOT / folder).glob("*.mp4")):
        rows.append({
            "clip": clip.name,
            "label": label,
            "path": clip.relative_to(ROOT).as_posix(),
            "species_note": "",
        })

df = pd.DataFrame(rows)

# Sanity checks
assert len(df) > 0, "No clips found. Check folder names and extension."
assert df["clip"].is_unique, "Duplicate filenames across folders!"
print(df["label"].value_counts())

Path("data").mkdir(exist_ok=True)
df.to_csv("data/labels.csv", index=False)