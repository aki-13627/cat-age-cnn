import os
import csv

# 先ほど整理したフォルダを指定
processed_dir = "data/processed"
csv_path = "data/filename-age.csv"

def main():
    # 1. 既存データの読み込み（重複回避のため）
    existing_filenames = set()
    if os.path.exists(csv_path):
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            try:
                # ヘッダーがある場合は読み飛ばす
                header = next(reader)
            except StopIteration:
                pass
            for row in reader:
                if row:
                    # CSVの1列目（ファイル名）をセットに登録
                    existing_filenames.add(row[0])

    # 2. 新規ファイルの収集
    new_rows = []
    
    if not os.path.exists(processed_dir):
        print(f"エラー: ディレクトリが見つかりません: {processed_dir}")
        return

    files = sorted(os.listdir(processed_dir))
    print(f"ディレクトリをスキャン中: {processed_dir}")

    for filename in files:
        # 画像ファイル以外はスキップ
        if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        # 既にCSVにあるファイルはスキップ
        if filename in existing_filenames:
            continue

        # ファイル名解析
        # 形式: Index(5桁)_Age(2桁)_Month(2桁)_Day(2桁)_Num.ext
        # 例: 00246_05_01_28_3.jpg -> split('_') -> ['00246', '05', '01', '28', '3.jpg']
        try:
            parts = filename.split('_')
            
            # 安全のためパーツ数を確認（最低でも Index, Age, Month, Day, Num の5つはあるはず）
            if len(parts) >= 5:
                # 2番目の要素(インデックス1)が年齢
                age = int(parts[1])
                new_rows.append([filename, age])
            else:
                # 形式が合わない場合はログを出してスキップ
                # print(f"スキップ（形式不一致）: {filename}")
                pass

        except ValueError:
            print(f"スキップ（数値変換エラー）: {filename}")
            continue

    # 3. 追記処理
    if new_rows:
        # ファイルが新規作成、または空の場合はヘッダーを書き込むフラグ
        write_header = not os.path.exists(csv_path) or os.stat(csv_path).st_size == 0

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            
            if write_header:
                writer.writerow(["filename", "age"])
                print("ヘッダーを作成しました。")

            for row in new_rows:
                writer.writerow(row)
                # 全て出力すると多い場合はコメントアウトしてください
                print(f"追記: {row}")
        
        print(f"完了: {len(new_rows)} 件のデータを追加しました。")
    else:
        print("追加する新しいデータはありませんでした。")

if __name__ == "__main__":
    main()