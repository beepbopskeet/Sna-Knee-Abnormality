# RSNA Knee Abnormality Detection

Kaggle "RSNA Knee Abnormality Detection" yarışması için multimodal (görüntü + rapor metni)
diz MR anormallik tespit modeli.

> ⚠️ **Bu proje bir araştırma/yarışma çalışmasıdır, tıbbi tanı aracı değildir.**
> Model çıktıları hiçbir koşulda gerçek hasta değerlendirmesi veya klinik karar için
> kullanılmamalıdır. Sonuçlar yalnızca teknik/eğitim amaçlıdır.

## Problem

- 12 diz anormalliği için per-study olasılık skoru üretmek (macro-average AUC-ROC ile
  değerlendiriliyor): ACL, MCL, Medial/Lateral Meniscus, Medial/Lateral/PF OA, Effusion,
  Synovitis, Baker's, Contusion, Fracture.
- Eğitim setinin sadece ~%1.3'ünde (58/4407 study) gerçek etiket var; geri kalanında sadece
  radyoloji raporu var. Bu yüzden **rapor metninden pseudo-label çıkarma** (weak supervision)
  ana strateji.
- Test aşamasında rapor metni verilmiyor — nihai model **sadece DICOM görüntülerinden**
  tahmin yapmak zorunda.
- Submission notebook'u internetsiz, ≤9 saat çalışmalı (Kaggle Code Competition kuralları).

## Pipeline

```
1. EDA                    -> notebooks/01_eda.py
2. Pseudo-labeling         -> src/pseudo_labeling.py   (rapor -> 12 etiket, altın sette doğrulanmış)
3. DICOM yükleme/önişleme  -> src/dicom_utils.py
4. Dataset / DataLoader    -> src/dataset.py
5. Model                   -> src/model.py             (multi-series aggregation CNN)
6. Eğitim                  -> src/train.py
7. Submission notebook     -> notebooks/kaggle_submission_notebook.ipynb
```

## Kurulum

```bash
pip install -r requirements.txt
```

## Kullanım sırası

```bash
# 1. Raporlardan pseudo-label üret (58 altın sette doğrulanmış kurallarla)
python src/pseudo_labeling.py --input data/train.csv --output data/pseudo_labels.csv

# 2. Eğitim (Kaggle GPU notebook'unda veya lokal GPU'da çalıştırılmalı — 570GB veri gerektirir)
python src/train.py --config configs/baseline.yaml

# 3. Kaggle'a submission notebook'u yükle
#    notebooks/kaggle_submission_notebook.ipynb dosyasını Kaggle Notebooks'a import et,
#    "Internet" kapalı olarak ayarla, eğitilmiş model ağırlıklarını dataset olarak ekle.
```

## Durum / Yapılacaklar

- [x] EDA tamamlandı (train.csv, train_series.csv analiz edildi)
- [x] Pseudo-labeling kuralları 58 altın örnekte doğrulandı (bkz. `EVAL_RESULTS.md`)
- [ ] Pseudo-labeling tüm veri setine (~4300 rapor) uygulanacak
- [ ] DICOM pipeline gerçek veriyle test edilecek (Kaggle ortamında)
- [ ] Baseline model eğitimi
- [ ] Submission notebook'unun Kaggle'da uçtan uca test edilmesi

## Lisans / Kurallar

Yarışma kurallarına tabidir: freely & publicly available external data/pretrained model
kullanımı serbest; LLM tabanlı pseudo-labeling yalnızca offline veri hazırlama aşamasında
kullanılmıştır (submission notebook'unda internet erişimi yoktur, bu adım orada tekrarlanmaz).
