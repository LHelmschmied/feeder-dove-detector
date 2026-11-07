from pathlib import Path
import pandas as pd

ROOT = Path("data/raw_clips")
FOLDER_TO_LABEL = {"dove_clips": "dove", "not_dove_clips": "other"}

rows = []
for folder, label in FOLDER_TO_LABEL.items():
    for clip in sorted((ROOT / folder).glob("*")):
        if clip.suffix.lower() == ".mp4":
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

# # Parse date/time from filenames like 2026-11-03_113243_113253_<id>.MP4
parts = df["clip"].str.extract(
    r"(\d{4}-\d{2}-\d{2})_(\d{2})\D?(\d{2})\D?(\d{2})_(\d{2})\D?(\d{2})\D?(\d{2})_"
)
bad = df.loc[parts.isna().any(axis=1), "clip"]
assert bad.empty, f"Filenames don't match the pattern:\n{bad.to_string()}"

start_t = parts[1] + parts[2] + parts[3]
end_t = parts[4] + parts[5] + parts[6]
df["date"] = pd.to_datetime(parts[0]).dt.date
df["start"] = pd.to_datetime(parts[0] + " " + start_t, format="%Y-%m-%d %H%M%S")
df["end"] = pd.to_datetime(parts[0] + " " + end_t, format="%Y-%m-%d %H%M%S")
df["duration_s"] = (df["end"] - df["start"]).dt.total_seconds()
df["hour"] = df["start"].dt.hour

print(df.groupby(["date", "label"]).size().unstack(fill_value=0))

Path("data").mkdir(exist_ok=True)
df.to_csv("data/labels.csv", index=False)