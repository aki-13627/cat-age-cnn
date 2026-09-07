import pandas as pd

def analyze_split_data(file_path):
    """
    CSVファイルから画像枚数とユニークな個体数をカウントします。
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません: {file_path}")
        return None

    # ====================================================
    # 1. 個体IDの抽出
    # ====================================================
    # ファイル名 "00097_12_..." の最初の "_" までをIDとする
    # 例: 00097_12_00_00_0.jpg -> 00097
    df['subject_id'] = df['filename'].apply(lambda x: x.split('_')[0])

    print(f"全データ行数: {len(df)}")
    print(f"全ユニーク個体数: {df['subject_id'].nunique()}")
    print("-" * 40)

    # ====================================================
    # 2. Splitごとの集計
    # ====================================================
    splits = ['train', 'val']
    stats = {}
    
    # IDの集合を保存して後で重複チェックに使う
    id_sets = {}

    for split_name in splits:
        # 該当splitのデータを抽出
        subset = df[df['split'] == split_name]
        
        # 画像枚数（行数）
        count_images = len(subset)
        
        # ユニーク個体数
        unique_ids = subset['subject_id'].unique()
        count_cats = len(unique_ids)
        
        id_sets[split_name] = set(unique_ids)

        stats[split_name] = {
            'images': count_images,
            'cats': count_cats
        }

    # ====================================================
    # 3. データリーク（重複）のチェック
    # ====================================================
    # TrainとValの両方に含まれている個体IDがあるか確認
    overlap = id_sets['train'].intersection(id_sets['val'])
    
    return stats, overlap

if __name__ == '__main__':
    csv_file_path = 'data/filename-age-split.csv'
    result = analyze_split_data(csv_file_path)

    if result:
        stats, overlap = result
        
        print(f"{'Split':<10} | {'画像枚数':<10} | {'個体数(ユニーク)':<15}")
        print("-" * 45)
        
        total_img = 0
        total_cats = 0
        
        for split_name, data in stats.items():
            print(f"{split_name:<10} | {data['images']:<10} | {data['cats']:<15}")
            total_img += data['images']
            total_cats += data['cats'] # 単純合計（重複があればここはずれる）

        print("-" * 45)
        print(f"{'Total':<10} | {total_img:<10} | {total_cats:<15} (単純合計)")

        print("\n--- データリークチェック ---")
        if len(overlap) > 0:
            print(f"⚠️ 警告: {len(overlap)} 匹の猫が Train と Val の両方に含まれています！")
            print(f"重複ID: {overlap}")
        else:
            print("✅ 正常: TrainとValで個体の重複はありません（完全に分離されています）。")