# =========================
# データリークチェック：　trainのidがvalに含まれているかをチェックする
# =========================

00147_04_00_00_0.jpg,4,train

CSV_FILE = "/Users/akihiro/cat-age-cnn/data/filename-age-split.csv"

import pandas as pd
def data_leak_check(csv_file):
    df = pd.read_csv(CSV_FILE)
    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    train_ids = train_df["filename"].unique()
    val_ids = val_df["filename"].unique()
    for id in train_ids:
        if id in val_ids:
            print(f"データリークが発生しています: {id}")
            return False
    return True

if __name__ == "__main__":
    result = data_leak_check(CSV_FILE)
    print(result)