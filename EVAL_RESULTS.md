# Pseudo-Labeling Değerlendirme Sonuçları

58 kişilik altın (ground-truth) sette, rapor metninden 12 etiket çıkarma performansı:

| Etiket            | Accuracy | F1   | Precision | Recall |
|-------------------|----------|------|-----------|--------|
| ACL               | 0.88     | 0.87 | 0.77      | 1.00   |
| MCL               | 0.86     | 0.69 | 0.53      | 1.00   |
| Medial Meniscus   | 0.86     | 0.85 | 0.82      | 0.88   |
| Lateral Meniscus  | 0.84     | 0.81 | 0.79      | 0.83   |
| Medial OA         | 0.91     | 0.85 | 0.78      | 0.93   |
| Lateral OA        | 0.86     | 0.69 | 0.60      | 0.82   |
| PF OA             | 0.84     | 0.74 | 0.93      | 0.62   |
| Effusion          | 0.72     | 0.81 | 0.69      | 1.00   |
| Synovitis         | 0.69     | 0.59 | 0.76      | 0.48   |
| Baker's           | 0.90     | 0.79 | 0.69      | 0.92   |
| Contusion         | 0.72     | 0.67 | 0.55      | 0.84   |
| Fracture          | 0.83     | 0.74 | 0.70      | 0.78   |

**Genel: micro-accuracy 0.828, macro-F1 0.759**

## Gözlemler ve alınan aksiyonlar

- **Synovitis (F1 0.59):** Precision yüksek, recall düşük — dolaylı ifadeler
  ("sinovyal kalınlaşma", "sinovyal hipertrofi", "reaktif sinovyalitis") kaçırılıyordu.
  → `pseudo_labeling.py` promptuna bu eşanlamlılar açıkça eklendi.
- **MCL / Contusion (F1 0.69 / 0.67):** Precision düşük — dejeneratif/kronik değişiklikler
  akut yaralanma ile karıştırılıyordu (örn. "entezopatik değişiklik" MCL yaralanması
  sayılmamalı; dejeneratif kemik iliği ödemi kontüzyon sayılmamalı).
  → Kural netleştirildi: sadece **akut/travmatik** bulgular sayılır, dejeneratif/kronik
  değişiklikler hariç tutulur.
- **Lateral OA (F1 0.69):** Kompartman-spesifik OA teşhisini genel ifadelerden ayırt etmek
  zordu ("trikompartmental" gibi genel terimler bazen yanlış kompartmana atfediliyordu).
  → Kural: yalnızca raporun **açıkça o kompartmanı** işaret ettiği durumlarda (veya impresyon/
  sonuç bölümünde "trikompartmental"/"her üç kompartman" dendiğinde) o kompartman pozitif
  sayılır.

## Sonraki adım

Bu iyileştirilmiş kurallarla `src/pseudo_labeling.py` tüm ~4300 etiketsiz rapora
uygulanacak. Çıktı `data/pseudo_labels.csv` olarak kaydedilip görüntü modelinin eğitim
etiketi olarak kullanılacak.
