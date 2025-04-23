import pandas as pd
from sklearn.model_selection import train_test_split

csv_path = "/Users/akihiro/cat-age-cnn/data/filename-age.csv"
df = pd.read_csv(csv_path)

# クラスごとの出現数を確認
class_counts = df["age"].value_counts()

# rareクラス（出現数1件未満）は train に固定
rare_classes = class_counts[class_counts < 2].index.tolist()
rare_df = df[df["age"].isin(rare_classes)]
main_df = df[~df["age"].isin(rare_classes)]

# stratified split → test = 10%
train_val_df, test_df = train_test_split(
    main_df, test_size=0.1, stratify=main_df["age"], random_state=42
)

# train:val = 70:20 → train = 0.7 / 0.9 ≒ 0.777..., val = 0.2 / 0.9 ≒ 0.222...
train_df, val_df = train_test_split(
    train_val_df,
    test_size=2/7,  # 約0.222
    stratify=train_val_df["age"],
    random_state=42,
)

# rare データを train に追加
train_df = pd.concat([train_df, rare_df]).reset_index(drop=True)

# split カラム追加
train_df["split"] = "train"
val_df["split"] = "val"
test_df["split"] = "test"

# マージ・保存
final_df = pd.concat([train_df, val_df, test_df]).reset_index(drop=True)
final_df = final_df.sort_values(by="filename")

final_df.to_csv("/Users/akihiro/cat-age-cnn/data/filename-age-split.csv", index=False, encoding="utf-8")
print("保存完了！ファイル: filename-age-split.csv")
