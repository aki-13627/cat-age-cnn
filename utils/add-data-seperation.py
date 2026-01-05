import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
import os

# パス設定
csv_path = "/Users/akihiro/cat-age-cnn/data/filename-age.csv"
output_path = "/Users/akihiro/cat-age-cnn/data/filename-age-split.csv"

def main():
    # 1. データの読み込み
    if not os.path.exists(csv_path):
        print(f"エラー: ファイルが見つかりません {csv_path}")
        return
        
    df = pd.read_csv(csv_path)

    # 2. 分割計算用に一時的な 'cat_index' 列を作成
    # ファイル名 "00246_05_01..." の先頭 "00246" を取得
    df["cat_index"] = df["filename"].apply(lambda x: x.split("_")[0])

    print(f"総画像数: {len(df)}")
    print(f"ユニークな猫の数: {df['cat_index'].nunique()}")

    # 3. グループ分割 (GroupShuffleSplit)
    # 同じIndexの猫がTrainとValに分かれないように分割
    # Testなし、Train:Val = 8:2 (test_size=0.2)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    
    train_idx, val_idx = next(gss.split(df, y=df["age"], groups=df["cat_index"]))

    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    # 4. splitラベルの付与
    train_df["split"] = "train"
    val_df["split"] = "val"

    # 5. データ結合
    final_df = pd.concat([train_df, val_df]).reset_index(drop=True)
    
    # 6. 不要なカラムの削除 (cat_index を削除)
    final_df = final_df.drop(columns=["cat_index"])

    # 7. 並び替えと保存
    final_df = final_df.sort_values(by="filename")
    final_df.to_csv(output_path, index=False, encoding="utf-8")
    
    # --- 確認用出力 ---
    print(f"\n保存完了！ファイル: {output_path}")
    print("-" * 30)
    print(f"Trainデータ数: {len(train_df)}")
    print(f"Valデータ数:   {len(val_df)}")
    print("-" * 30)
    print("保存されたカラム:")
    print(final_df.columns.tolist())

if __name__ == "__main__":
    main()