import glob
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

folds = pd.read_csv("data/folds.csv", dtype={"clip_id": str})

def scores(y, pred):
    p, r, f1, _ = precision_recall_fscore_support(y, pred, pos_label=1,
                                                  average="binary", zero_division=0)
    return p, r, f1

def summarize(name, rows):
    a = np.array(rows)
    print(f"{name:22s} precision {a[:,0].mean():.2f}±{a[:,0].std():.2f} | "
          f"recall {a[:,1].mean():.2f}±{a[:,1].std():.2f} | "
          f"F1 {a[:,2].mean():.2f}±{a[:,2].std():.2f}")

# Image models: read saved per-clip predictions
for suffix, name in [("", "frozen ResNet18"), ("_unfrozen", "layer4 unfrozen")]:
    rows = []
    for k in range(5):
        try:
            c = pd.read_csv(f"results/preds_fold{k}{suffix}.csv")
        except FileNotFoundError:
            continue
        rows.append(scores(c["y"], c["pred"]))
    if rows:
        summarize(name, rows)

# Time-only baseline: per-hour dove rate learned from the training folds
rows = []
for k in range(5):
    tr, va = folds[folds["fold"] != k], folds[folds["fold"] == k]
    rate = (tr["label"] == "dove").groupby(tr["hour"]).mean()
    prior = (tr["label"] == "dove").mean()
    p_dove = va["hour"].map(rate).fillna(prior)
    rows.append(scores((va["label"] == "dove").astype(int), (p_dove >= 0.5).astype(int)))
summarize("time-only (hour)", rows)

# Majority-class baseline is always "other": dove F1 = 0