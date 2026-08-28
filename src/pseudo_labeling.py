"""
Rapor metninden 12 diz anormalliği etiketi çıkarma (weak supervision / pseudo-labeling).

ÖNEMLİ: Bu script yalnızca OFFLINE veri hazırlama için kullanılır. Kaggle submission
notebook'unda internet erişimi kapalı olduğu için bu script'in çıktısı (pseudo_labels.csv)
önceden üretilip eğitim verisine dahil edilir; submission notebook'u içinde ASLA
çağrılmaz / çağrılamaz.

Kullanım:
    export ANTHROPIC_API_KEY=sk-...
    python src/pseudo_labeling.py --input data/train.csv --output data/pseudo_labels.csv
    python src/pseudo_labeling.py --input data/train.csv --output data/gold_check.csv --gold-only
"""
import argparse
import json
import os
import time
import sys

import pandas as pd
import requests

LABEL_COLS = ['ACL', 'MCL', 'Medial Meniscus', 'Lateral Meniscus', 'Medial OA',
              'Lateral OA', 'PF OA', 'Effusion', 'Synovitis', "Baker's",
              'Contusion', 'Fracture']

# 58 altın örnek üzerindeki hata analizinden (bkz. EVAL_RESULTS.md) güncellenmiş kurallar.
SYSTEM_PROMPT = """Sen bir kas-iskelet radyoloğusun. Sana bir diz MR radyoloji raporu verilecek.
Rapor İngilizce, İspanyolca, Türkçe, Almanca, Hollandaca, Fransızca, Yunanca, Hırvatça veya
Bulgarca dillerinden birinde olabilir.

Görevin: Raporda aşağıdaki 12 bulgudan hangilerinin var olduğunu tespit etmek.

Bulgular ve tanımları:
- ACL: ön çapraz bağ yaralanması/yırtığı/rüptürü (kısmi veya tam)
- MCL: iç yan bağ AKUT yaralanması/yırtığı/sprain'i (grade 1-3 dahil).
       Kronik entezopati, sadece kalınlaşma gibi dejeneratif bulgular SAYILMAZ.
- Medial Meniscus: iç menisküs yırtığı (radial, horizontal, kompleks, bucket-handle vb.)
       Sadece "dejenerasyon" veya "sinyal artışı, yüzeye ulaşmıyor" ifadeleri TEK BAŞINA
       yırtık sayılmaz; açıkça "tear/rotura/yırtık/ruptür/rupture" kelimesi veya yüzeye
       ulaşan sinyal değişikliği gerekir.
- Lateral Meniscus: dış menisküs yırtığı (yukarıdaki kural aynı şekilde geçerli)
- Medial OA: iç (medial) tibiofemoral kompartman osteoartriti. Açıkça "osteoarthritis/OA/
       artrosis/artroz/gonartroz" kelimesi kullanılmışsa VEYA o kompartmanda grade 2+
       kıkırdak kaybı + osteofit birlikte belirtilmişse pozitif say. Sadece hafif/grade 1
       kondromalazi TEK BAŞINA yeterli değildir.
- Lateral OA: dış (lateral) tibiofemoral kompartman osteoartriti (aynı kural)
- PF OA: patellofemoral kompartman osteoartriti (aynı kural)
- Effusion: eklem efüzyonu / sıvı artışı (hafif dahil, "trace" hariç tutulabilir ama "mild/
       leve/hafif" ve üzeri sayılır)
- Synovitis: sinovyal iltihap. "Synovitis/sinovit/sinovyalitis" kelimesi VEYA "sinovyal
       kalınlaşma", "sinovyal hipertrofi", "reaktif sinovyalitis", "hypertrophy of the
       synovium" gibi eşdeğer ifadeler de pozitif sayılır.
- Baker's: Baker kisti / popliteal kist (boyutu ne olursa olsun, "trace" dahil)
- Contusion: AKUT kemik kontüzyonu / travmatik kemik iliği ödemi ("bone bruise", "bone
       contusion", "impaction injury" ile ilişkili akut ödem). Dejeneratif/kronik OA'ya
       bağlı subkondral ödem SAYILMAZ (bu, ilgili OA etiketine dahildir, ayrı contusion
       değildir).
- Fracture: kırık (akut, stres, subkondral yetmezlik kırığı, avülsiyon kırığı dahil).
       "Impaction fracture", "insufficiency fracture", "avulsion fracture", "hairline
       fracture" hepsi sayılır.

KURALLAR:
1. Sadece raporda AÇIKÇA belirtilen veya güçlü şekilde ima edilen ("suspect", "favor of",
   "consistent with", "C/W", "R/O" + sonrasında teyit) bulguları pozitif (1) say.
2. Bahsedilmeyen veya açıkça normal/negatif olarak belirtilen bulguları 0 say.
3. Sadece geçerli JSON döndür, başka hiçbir açıklama ekleme. Format:
{"ACL":0,"MCL":0,"Medial Meniscus":0,"Lateral Meniscus":0,"Medial OA":0,"Lateral OA":0,"PF OA":0,"Effusion":0,"Synovitis":0,"Baker's":0,"Contusion":0,"Fracture":0}
"""

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-6"


def call_claude(report_text: str, api_key: str) -> dict:
    resp = requests.post(
        API_URL,
        headers={
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
        },
        json={
            "model": MODEL,
            "max_tokens": 300,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": f"Rapor:\n{report_text}\n\nJSON etiketleri ver:"}],
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").lstrip("json").strip()
    return json.loads(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="train.csv yolu")
    ap.add_argument("--output", required=True, help="çıktı csv yolu")
    ap.add_argument("--gold-only", action="store_true",
                     help="sadece gerçek etiketi olan (doğrulama için) satırları işle")
    ap.add_argument("--limit", type=int, default=None, help="test için ilk N satırla sınırla")
    ap.add_argument("--sleep", type=float, default=0.3, help="istekler arası bekleme (rate limit)")
    args = ap.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit("HATA: ANTHROPIC_API_KEY ortam değişkeni tanımlı değil.")

    df = pd.read_csv(args.input)
    if args.gold_only:
        df = df[df[LABEL_COLS].notna().all(axis=1)].reset_index(drop=True)
    if args.limit:
        df = df.head(args.limit)

    print(f"{len(df)} rapor işlenecek...")
    out_rows = []
    for i, row in df.iterrows():
        try:
            pred = call_claude(row["Report"], api_key)
        except Exception as e:
            print(f"  [{i}] HATA: {e} -> tüm etiketler 0 olarak kaydedildi", file=sys.stderr)
            pred = {c: 0 for c in LABEL_COLS}
        pred["StudyInstanceUID"] = row["StudyInstanceUID"]
        out_rows.append(pred)
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(df)} işlendi")
        time.sleep(args.sleep)

    out_df = pd.DataFrame(out_rows)[["StudyInstanceUID"] + LABEL_COLS]
    out_df.to_csv(args.output, index=False)
    print(f"Kaydedildi: {args.output} ({out_df.shape})")


if __name__ == "__main__":
    main()
