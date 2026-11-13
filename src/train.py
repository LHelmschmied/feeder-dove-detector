import argparse, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch, torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import FeederFrames

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)

def build_model(unfreeze_layer4=False):
    m = models.resnet18(weights="DEFAULT")
    for p in m.parameters():
        p.requires_grad = False
    if unfreeze_layer4:
        for p in m.layer4.parameters():
            p.requires_grad = True
    m.fc = nn.Linear(m.fc.in_features, 2)      # new head, trainable
    return m

def run_epoch(model, loader, criterion, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, n = 0.0, 0
    clip_ids, probs, ys = [], [], []
    with torch.set_grad_enabled(training):
        for x, y, cid in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y); n += len(y)
            probs += torch.softmax(out, 1)[:, 1].detach().cpu().tolist()
            ys += y.cpu().tolist(); clip_ids += list(cid)
    return total_loss / n, pd.DataFrame({"clip_id": clip_ids, "p_dove": probs, "y": ys})

def clip_metrics(pred_df):
    """Average frame probabilities per clip, then threshold at 0.5."""
    c = pred_df.groupby("clip_id").agg(p_dove=("p_dove", "mean"), y=("y", "first"))
    c["pred"] = (c["p_dove"] >= 0.5).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(c["y"], c["pred"], pos_label=1,
                                                  average="binary", zero_division=0)
    return p, r, f1, confusion_matrix(c["y"], c["pred"], labels=[0, 1]), c

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--unfreeze", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    set_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = FeederFrames(a.fold, train=True)
    val_ds = FeederFrames(a.fold, train=False)
    train_dl = DataLoader(train_ds, batch_size=32, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=32, shuffle=False, num_workers=0)
    print(f"fold {a.fold}: {len(train_ds)} train frames, {len(val_ds)} val frames, device={device}")

    # class weights: inverse frequency (computed on training frames only)
    counts = train_ds.df["label"].value_counts()
    w = torch.tensor([1 / counts["other"], 1 / counts["dove"]], dtype=torch.float)
    w = (w / w.sum() * 2).to(device)
    criterion = nn.CrossEntropyLoss(weight=w)

    model = build_model(a.unfreeze).to(device)
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=a.lr)

    hist = {"train": [], "val": []}
    for ep in range(1, a.epochs + 1):
        tr_loss, _ = run_epoch(model, train_dl, criterion, device, optimizer)
        va_loss, va_pred = run_epoch(model, val_dl, criterion, device)
        _, _, f1, _, _ = clip_metrics(va_pred)
        hist["train"].append(tr_loss); hist["val"].append(va_loss)
        print(f"epoch {ep:2d} | train loss {tr_loss:.3f} | val loss {va_loss:.3f} | clip F1 {f1:.2f}")

    p, r, f1, cm, clips = clip_metrics(va_pred)
    print(f"\nFold {a.fold} clip-level dove precision {p:.2f} recall {r:.2f} F1 {f1:.2f}")
    print("confusion matrix [[TN FP],[FN TP]]:\n", cm)

    Path("results").mkdir(exist_ok=True)
    tag = f"fold{a.fold}{'_unfrozen' if a.unfreeze else ''}"
    clips.to_csv(f"results/preds_{tag}.csv")
    plt.plot(hist["train"], label="train"); plt.plot(hist["val"], label="val")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend()
    plt.savefig(f"results/loss_{tag}.png", dpi=150)

if __name__ == "__main__":
    main()