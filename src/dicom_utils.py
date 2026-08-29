"""
DICOM okuma ve önişleme yardımcı fonksiyonları.

Veri yapısı: train_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm
Her seri 20-45 slice (medyan 30). Transfer syntax karışık (uncompressed, JPEG Lossless,
JPEG 2000, Implicit VR) -- pylibjpeg veya gdcm eklentisi gerekebilir, bkz. requirements.txt.
"""
import os
from pathlib import Path

import numpy as np
import cv2

try:
    import pydicom
except ImportError:
    pydicom = None


def list_series_files(study_dir: str, series_uid: str) -> list:
    """Bir serideki tüm .dcm dosyalarını InstanceNumber'a göre sıralı döndürür."""
    series_dir = Path(study_dir) / series_uid
    files = sorted(series_dir.glob("*.dcm"))
    return [str(f) for f in files]


def read_dicom_slice(path: str, resize_to: int = None) -> np.ndarray:
    """Tek bir DICOM slice'ı okuyup normalize edilmiş float32 array olarak döndürür.
    resize_to verilirse, çıktı (resize_to, resize_to) boyutuna getirilir -- farklı
    serilerin farklı native çözünürlükte olması nedeniyle bu adım ZORUNLUDUR, yoksa
    np.stack ile birleştirirken 'all input arrays must have the same shape' hatası alınır.
    """
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)

    # Rescale slope/intercept uygula (varsa)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    arr = arr * slope + intercept

    # Min-max normalize [0, 1]
    lo, hi = np.percentile(arr, [0.5, 99.5])
    arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)

    if resize_to is not None and arr.shape != (resize_to, resize_to):
        arr = cv2.resize(arr, (resize_to, resize_to), interpolation=cv2.INTER_AREA)

    return arr


def load_series_volume(study_dir: str, series_uid: str, target_slices: int = 24,
                        resize_to: int = 224) -> np.ndarray:
    """
    Bir seriyi sabit sayıda slice'a örnekleyip (target_slices) ve her slice'ı
    resize_to x resize_to boyutuna getirip (T, H, W) array döndürür.

    Not: cv2/PIL resize burada bilerek dışarıda bırakıldı -- gerçek ortamda
    `cv2.resize` veya `torchvision.transforms` kullanılmalı.
    """
    files = list_series_files(study_dir, series_uid)
    if not files:
        return np.zeros((target_slices, resize_to, resize_to), dtype=np.float32)

    # Eşit aralıklarla target_slices kadar slice seç
    idx = np.linspace(0, len(files) - 1, target_slices).astype(int)
    slices = []
    for i in idx:
        try:
            arr = read_dicom_slice(files[i], resize_to=resize_to)
        except Exception:
            arr = np.zeros((resize_to, resize_to), dtype=np.float32)
        slices.append(arr)
    return np.stack(slices, axis=0)


def get_study_series_uids(series_df, study_uid: str, plane: str = None,
                           fluid_sensitive: int = None) -> list:
    """
    train_series.csv üzerinden bir study'ye ait seri ID'lerini filtreler.
    plane: 'Sagittal' | 'Coronal' | 'Axial' | None (hepsi)
    fluid_sensitive: 0 | 1 | None (hepsi)
    """
    mask = series_df["StudyInstanceUID"] == study_uid
    if plane is not None:
        mask &= series_df["Anatomical_Plane"] == plane
    if fluid_sensitive is not None:
        mask &= series_df["Fluid_Sensitive"] == fluid_sensitive
    return series_df.loc[mask, "SeriesInstanceUID"].tolist()
