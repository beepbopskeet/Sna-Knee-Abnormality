"""
Study bazlı multi-series MRI Dataset.

Önceki versiyon her plandan (Sagittal/Coronal/Axial) sadece 1 seri seçiyordu. Bu
versiyon bir study'ye ait TÜM serileri (max_views'e kadar) kullanır -- model.py'deki
attention pooling bu değişken sayıdaki "görünüm"ü (view) tek bir study embedding'ine
indirger. Böylece study'de birden fazla seri olan (örn. 2 sagittal farklı sekans)
durumlarda bilgi kaybı olmaz.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dicom_utils import load_series_volume, get_study_series_uids

LABEL_COLS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
              'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
              'Contusion', 'Fracture']

MAX_VIEWS = 6  # bir study'den en fazla kaç seri kullanılacağı (median 5.5 idi; 8 denendi,
                # düşük değerli seriler gürültü kattı, AUC düştü -- 6'ya çekildi)


class KneeMRIDataset(Dataset):
    def __init__(self, labels_df: pd.DataFrame, series_df: pd.DataFrame, series_root: str,
                 target_slices: int = 12, resize_to: int = 192, is_test: bool = False,
                 max_views: int = MAX_VIEWS):
        """
        labels_df: StudyInstanceUID + (varsa) 12 etiket kolonu (train: pseudo/gold label, test: yok)
        series_df: train_series.csv / test_series.csv
        series_root: train_series/ veya test_series/ klasör yolu
        """
        self.labels_df = labels_df.reset_index(drop=True)
        self.series_df = series_df
        self.series_root = series_root
        self.target_slices = target_slices
        self.resize_to = resize_to
        self.is_test = is_test
        self.max_views = max_views

    def __len__(self):
        return len(self.labels_df)

    def _load_study_views(self, study_uid: str) -> tuple:
        """Study'ye ait serileri ÖNCELİK SIRASINA göre (Fluid_Sensitive=1 önce, sonra
        diğerleri) yükler, max_views'e kadar. Öncelik sırası önemli: lokalizasyon/referans
        gibi tanısal değeri düşük serileri elemeden hepsini kullanmak modele gürültü katıp
        performansı düşürebiliyor (v1 denemesinde gözlemlendi) -- bu yüzden en bilgilendirici
        seriler (fluid-sensitive sekanslar) öncelikli sırada, kırpma olursa önce onlar kalır.
        Dönüş: (views, mask) -- views: (max_views, target_slices, H, W), mask: (max_views,)."""
        mask_rows = self.series_df["StudyInstanceUID"] == study_uid
        study_series = self.series_df.loc[mask_rows].copy()
        if len(study_series) == 0:
            series_ids = []
        else:
            # Fluid_Sensitive=1 olanlar önce gelsin (daha bilgilendirici sekanslar)
            study_series = study_series.sort_values("Fluid_Sensitive", ascending=False)
            series_ids = study_series["SeriesInstanceUID"].tolist()[:self.max_views]

        views = []
        for sid in series_ids:
            vol = load_series_volume(f"{self.series_root}/{study_uid}", sid,
                                      target_slices=self.target_slices,
                                      resize_to=self.resize_to)
            views.append(vol)

        n_real = len(views)
        while len(views) < self.max_views:
            views.append(np.zeros((self.target_slices, self.resize_to, self.resize_to),
                                   dtype=np.float32))

        mask = np.zeros(self.max_views, dtype=np.float32)
        mask[:n_real] = 1.0
        if n_real == 0:
            mask[0] = 1.0  # tamamen boşsa bile en az 1 (sıfır) görünüm işaretli olsun, NaN'dan kaçın

        return np.stack(views, axis=0), mask

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        study_uid = row["StudyInstanceUID"]
        views, mask = self._load_study_views(study_uid)
        views = torch.from_numpy(views).float()   # (max_views, T, H, W)
        mask = torch.from_numpy(mask).float()      # (max_views,)

        if self.is_test:
            return views, mask, study_uid

        labels = torch.tensor(row[LABEL_COLS].astype(float).values, dtype=torch.float32)
        return views, mask, labels
