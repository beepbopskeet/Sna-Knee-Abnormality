"""
Her study için DICOM header'larından "site/cihaz" grubu çıkarır (Manufacturer +
ManufacturerModelName + MagneticFieldStrength birleşimi, proxy olarak). Bu,
GroupKFold ile aynı hastane/cihazdan gelen görüntülerin train/val'e karışmasını
(ve modelin hastalığı değil "hangi cihaz" olduğunu ezberlemesini) engellemek içindir.

Bu script Kaggle'da (CPU yeterli, GPU gerekmez) train_series/ klasörüne erişimi
olan bir notebook'ta çalıştırılmalıdır -- yerel ortamda çalıştırılamaz (DICOM
dosyaları burada yok).

Kullanım (Kaggle):
    !python src/extract_site_groups.py \
        --series_csv /kaggle/input/competitions/rsna-knee-abnormality-detection/train_series.csv \
        --series_root /kaggle/input/competitions/rsna-knee-abnormality-detection/train_series \
        --output data/site_groups.csv
"""
import argparse
import os

import pandas as pd
import pydicom
from tqdm import tqdm


def get_site_key(study_dir: str, series_uid: str) -> str:
    """Bir serideki ilk DICOM dosyasından cihaz/site proxy anahtarı çıkarır."""
    series_path = os.path.join(study_dir, series_uid)
    if not os.path.isdir(series_path):
        return "unknown"
    files = sorted(os.listdir(series_path))
    if not files:
        return "unknown"
    try:
        ds = pydicom.dcmread(os.path.join(series_path, files[0]), stop_before_pixels=True)
        manufacturer = str(getattr(ds, "Manufacturer", "unk")).strip()
        model = str(getattr(ds, "ManufacturerModelName", "unk")).strip()
        field = str(getattr(ds, "MagneticFieldStrength", "unk")).strip()
        return f"{manufacturer}|{model}|{field}"
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series_csv", required=True)
    ap.add_argument("--series_root", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    series_df = pd.read_csv(args.series_csv)
    study_ids = series_df["StudyInstanceUID"].unique()

    rows = []
    for study_uid in tqdm(study_ids, desc="Site metadata cikariliyor"):
        study_series = series_df[series_df["StudyInstanceUID"] == study_uid]
        first_series = study_series.iloc[0]["SeriesInstanceUID"]
        study_dir = os.path.join(args.series_root, study_uid)
        site_key = get_site_key(study_dir, first_series)
        rows.append({"StudyInstanceUID": study_uid, "site_group": site_key})

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.output, index=False)
    print(f"Kaydedildi: {args.output} ({out_df.shape})")
    print(f"Benzersiz site/cihaz grubu sayisi: {out_df['site_group'].nunique()}")
    print(out_df["site_group"].value_counts().head(10))


if __name__ == "__main__":
    main()
