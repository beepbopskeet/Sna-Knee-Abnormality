"""
Study bazlı multi-series MRI Dataset.

Her study'de ortalama 5.5 seri (Sagittal/Coronal/Axial, farklı sekanslar) var. Strateji:
her plandan (Sagittal, Coronal, Axial) bir seri seç (Fluid_Sensitive tercih edilerek),
her birinden target_slices kadar slice örnekle, modele "çoklu görünüm" (multi-view) olarak ver.
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from dicom_utils import load_series_volume, get_study_series_uids

LABEL_COLS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
              'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
              'Contusion', 'Fracture']

PLANES = ["Sagittal", "Coronal", "Axial"]


class KneeMRIDataset(Dataset):
    def __init__(self, labels_df: pd.DataFrame, series_df: pd.DataFrame, series_root: str,
                 target_slices: int = 24, resize_to: int = 224, is_test: bool = False):
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

    def __len__(self):
        return len(self.labels_df)

    def _load_study_views(self, study_uid: str) -> np.ndarray:
        """Her plan için bir seri yükler; eksik plan varsa sıfır doldurur.
        Dönüş: (n_planes=3, target_slices, H, W)"""
        views = []
        for plane in PLANES:
            series_ids = get_study_series_uids(self.series_df, study_uid, plane=plane,
                                                fluid_sensitive=1)
            if not series_ids:
                series_ids = get_study_series_uids(self.series_df, study_uid, plane=plane)
            if series_ids:
                vol = load_series_volume(f"{self.series_root}/{study_uid}", series_ids[0],
                                          target_slices=self.target_slices,
                                          resize_to=self.resize_to)
            else:
                vol = np.zeros((self.target_slices, self.resize_to, self.resize_to),
                                dtype=np.float32)
            views.append(vol)
        return np.stack(views, axis=0)

    def __getitem__(self, idx):
        row = self.labels_df.iloc[idx]
        study_uid = row["StudyInstanceUID"]
        views = self._load_study_views(study_uid)
        views = torch.from_numpy(views).float()  # (3, T, H, W)

        if self.is_test:
            return views, study_uid

        labels = torch.tensor(row[LABEL_COLS].astype(float).values, dtype=torch.float32)
        return views, labels
