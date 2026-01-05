import pandas as pd

def count_split_data(file_path):
    """
    指定されたCSVファイルからtrain, val, testのデータ数をカウントします。

    Args:
        file_path (str): CSVファイルのパス。
    
    Returns:
        dict: 各分割のデータ数を格納した辞書。
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        print(f"エラー: ファイルが見つかりません。パスを確認してください: {file_path}")
        return None

    total_rows = len(df)
    print(f"全データ数: {total_rows}")

    split_counts = df['split'].value_counts().to_dict()
    return split_counts

if __name__ == '__main__':
    csv_file_path = 'data/filename-age-split.csv'
    counts = count_split_data(csv_file_path)

    if counts:
        print("\n各データ分割の行数:")
        print(f"train: {counts.get('train', 0)}")
        print(f"val:   {counts.get('val', 0)}")