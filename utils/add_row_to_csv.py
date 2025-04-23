import os
import csv
import re

processed_dir = "data/processed-for-cnn"
csv_path = "age_data.csv"

def clean_filename(fname: str) -> str:
    # 不要な .（ドット）を削除（例: 00101_00_00_09._face_0.jpg → 00101_00_00_09_face_0.jpg）
    fname = re.sub(r"\.(?=_face_)", "", fname)
    # 2重拡張子を修正（例: .jpg_face_0.jpg → _face_0.jpg）
    fname = re.sub(r"\.jpg(?=_face_)", "", fname)
    return fname.strip()

# 既にCSVにあるファイル名を読み込み（正規化付き）
existing_filenames = set()
if os.path.exists(csv_path):
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)  # ヘッダー
        for row in reader:
            existing_filenames.add(clean_filename(row[0].lower()))

# 新規ファイルをスキャンして追記候補を収集
new_rows = []
for raw_fname in os.listdir(processed_dir):
    if not raw_fname.lower().endswith(".jpg") or "_face_" not in raw_fname:
        continue

    fname = clean_filename(raw_fname)
    fname_lower = fname.lower()

    if not fname_lower.startswith("00") or fname_lower in existing_filenames:
        continue

    try:
        age = int(fname.split("_")[1])  # 例: 00034_10_00_00_face_0.jpg → age=10
        new_rows.append([fname, age])
    except (IndexError, ValueError) as e:
        print(f"無効なファイル名形式: {raw_fname} → スキップします ({e})")

# 追記処理
if new_rows:
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        for row in new_rows:
            writer.writerow(row)
            print(f"追記しました: {row}")
else:
    print("新しいデータはありませんでした。")
