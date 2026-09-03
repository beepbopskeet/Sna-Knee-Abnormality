"""
Eğitim döngüsü. Kaggle GPU notebook'unda veya lokal GPU'da çalıştırılmalıdır
(570GB DICOM verisi + GPU gerektirir -- bu ortamda çalıştırılamaz).

Kullanım:
    python src/train.py --config configs/baseline.yaml
"""
import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, GroupKFold
from torch.utils.data import DataLoader

from dataset import KneeMRIDataset, LABEL_COLS
from model import KneeMultiViewModel


def macro_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Yarışma metriği: 12 etiket üzerinde ortalama AUC-ROC."""
    scores = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) < 2:
            continue  # o fold'da tek sınıf varsa AUC tanımsız, atla
        scores.append(roc_auc_score(y_true[:, i], y_pred[:, i]))
    return float(np.mean(scores))


def train_one_fold(cfg, train_df, val_df, series_df, fold: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(cfg["output_dir"], exist_ok=True)

    train_ds = KneeMRIDataset(train_df, series_df, cfg["series_root"],
                               target_slices=cfg["target_slices"], resize_to=cfg["resize_to"])
    val_ds = KneeMRIDataset(val_df, series_df, cfg["series_root"],
                             target_slices=cfg["target_slices"], resize_to=cfg["resize_to"])

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                               num_workers=cfg["num_workers"], pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["num_workers"], pin_memory=True)

    model = KneeMultiViewModel(backbone_name=cfg["backbone"], pretrained=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    criterion = nn.BCEWithLogitsLoss()

    best_auc = 0.0
    for epoch in range(cfg["epochs"]):
        model.train()
        for views, labels in train_loader:
            views, labels = views.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(views)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                continue  # NaN/inf kaybı veren batch'i atla, eğitimi bozmasın
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step()

        model.eval()
        all_true, all_pred = [], []
        with torch.no_grad():
            for views, labels in val_loader:
                views = views.to(device)
                logits = model(views)
                probs = torch.sigmoid(logits).float().cpu().numpy()
                probs = np.nan_to_num(probs, nan=0.5)  # kalan olası NaN'ları güvenli değere çevir
                all_pred.append(probs)
                all_true.append(labels.numpy())
        y_true = np.concatenate(all_true)
        y_pred = np.concatenate(all_pred)
        auc = macro_auc(y_true, y_pred)
        print(f"[fold {fold}] epoch {epoch + 1}/{cfg['epochs']}  val macro-AUC={auc:.4f}")

        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), f"{cfg['output_dir']}/model_fold{fold}.pt")

    return best_auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    labels_df = pd.read_csv(cfg["pseudo_labels_csv"])
    series_df = pd.read_csv(cfg["train_series_csv"])

    # Site/cihaz bazlı gruplu fold: aynı hastane/cihazdan gelen study'ler hep aynı
    # fold'da kalır, böylece model "hangi cihaz" yerine gerçek patolojiyi öğrenmeye
    # zorlanır (bkz. README -- rastgele fold, AUC'yi yapay olarak şişirebiliyor).
    site_groups_path = cfg.get("site_groups_csv")
    if site_groups_path and os.path.exists(site_groups_path):
        site_df = pd.read_csv(site_groups_path)
        labels_df = labels_df.merge(site_df, on="StudyInstanceUID", how="left")
        labels_df["site_group"] = labels_df["site_group"].fillna("unknown")
        groups = labels_df["site_group"].values
        kf = GroupKFold(n_splits=cfg["n_folds"])
        split_iter = kf.split(labels_df, groups=groups)
        print(f"GroupKFold kullaniliyor ({labels_df['site_group'].nunique()} benzersiz grup).")
    else:
        kf = KFold(n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["seed"])
        split_iter = kf.split(labels_df)
        print("UYARI: site_groups_csv bulunamadi, rastgele KFold kullaniliyor "
              "(bkz. README -- GroupKFold icin once extract_site_groups.py calistirin).")

    fold_scores = []
    for fold, (train_idx, val_idx) in enumerate(split_iter):
        train_df = labels_df.iloc[train_idx]
        val_df = labels_df.iloc[val_idx]
        auc = train_one_fold(cfg, train_df, val_df, series_df, fold)
        fold_scores.append(auc)

    print(f"\nOrtalama CV macro-AUC: {np.mean(fold_scores):.4f} (+/- {np.std(fold_scores):.4f})")


if __name__ == "__main__":
    main()
