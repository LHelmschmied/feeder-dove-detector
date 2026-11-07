from pathlib import Path
import cv2
import numpy as np
import pandas as pd

ROOT = Path("data/raw_clips")
OUT = Path("data/frames")
N_FRAMES = 8
SHORT_SIDE = 256

df = pd.read_csv("data/labels.csv")
OUT.mkdir(parents=True, exist_ok=True)

rows = []
for _, r in df.iterrows():
    cap = cv2.VideoCapture(str(ROOT / r["path"]))
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    if len(frames) == 0:
        print("Could not read", r["clip_id"])
        continue

    idxs = np.linspace(0, len(frames) - 1, N_FRAMES + 2)[1:-1].astype(int)  # skip first/last
    for k, i in enumerate(idxs):
        f = frames[i].copy()
        h, w = f.shape[:2]
        f[: int(0.09 * h), int(0.78 * w):] = 128      # timestamp
        f[int(0.90 * h):, : int(0.16 * w)] = 128      # "Bird feeder" label
        scale = SHORT_SIDE / min(h, w)
        f = cv2.resize(f, (int(w * scale), int(h * scale)))
        name = f"{r['clip_id']}_f{k}.jpg"
        cv2.imwrite(str(OUT / name), f)
        rows.append({"clip_id": r["clip_id"], "frame_file": name, "frame_idx": int(i),
                     "label": r["label"], "hour": r["hour"]})
    print("done", r["clip_id"])

pd.DataFrame(rows).to_csv("data/frames.csv", index=False)
print(len(rows), "frames saved")