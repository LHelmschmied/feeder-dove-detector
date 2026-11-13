import pandas as pd
from sklearn.model_selection import StratifiedKFold

N_FOLDS = 5
SEED = 42

df = pd.read_csv("data/labels.csv", dtype={"clip_id": str})
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
df["fold"] = -1
for k, (_, val_idx) in enumerate(skf.split(df, df["label"])):
    df.loc[val_idx, "fold"] = k

print(pd.crosstab(df["fold"], df["label"]))
df[["clip_id", "label", "hour", "fold"]].to_csv("data/folds.csv", index=False)