import pandas as pd
import os
import sys

# ==========================================
# ★設定
# ==========================================
# 入力CSVファイルのパス
CSV_PATH = 'data/filename-age-split.csv'

# 出力ログファイルのパス
LOG_PATH = 'outputs/dataset_age_distribution.log'

# ==========================================
# メイン処理
# ==========================================
def main():
    # 1. CSVファイルの読み込み
    if not os.path.exists(CSV_PATH):
        print(f"エラー: ファイルが見つかりません -> {CSV_PATH}")
        sys.exit(1)

    try:
        df = pd.read_csv(CSV_PATH)
        print(f"CSVを読み込みました: {len(df)} 行")
    except Exception as e:
        print(f"CSV読み込みエラー: {e}")
        sys.exit(1)

    # 2. 集計処理 (クロス集計)
    # index=age, columns=split で集計
    summary = pd.crosstab(df['age'], df['split'])

    # train または val の列が存在しない場合（片方しかないデータセットの場合など）の対応
    if 'train' not in summary.columns:
        summary['train'] = 0
    if 'val' not in summary.columns:
        summary['val'] = 0

    # NaNを0で埋めて整数型にする
    summary = summary.fillna(0).astype(int)

    # 合計列を追加
    summary['total'] = summary['train'] + summary['val']

    # 年齢順にソート (通常はされていますが念のため)
    summary = summary.sort_index()

    # 3. ログ出力用のテキスト作成 (Markdownテーブル形式)
    log_lines = []
    log_lines.append(f"Dataset Analysis: {CSV_PATH}")
    log_lines.append(f"Date: {pd.Timestamp.now()}")
    log_lines.append("-" * 50)
    
    # ヘッダー
    log_lines.append("| Age | Train | Val | Total |")
    log_lines.append("|---|---|---|---|")

    # 各行のデータ
    for age, row in summary.iterrows():
        log_lines.append(f"| {age} | {row['train']} | {row['val']} | {row['total']} |")

    # 全体集計
    total_train = summary['train'].sum()
    total_val = summary['val'].sum()
    total_all = summary['total'].sum()

    log_lines.append("-" * 50)
    log_lines.append("### Summary Statistics")
    log_lines.append(f"Total Images : {total_all}")
    log_lines.append(f"Train Set    : {total_train} ({total_train/total_all*100:.1f}%)")
    log_lines.append(f"Val Set      : {total_val} ({total_val/total_all*100:.1f}%)")
    log_lines.append(f"Age Range    : {summary.index.min()} - {summary.index.max()} 歳")
    log_lines.append(f"Num Classes  : {len(summary)} クラス")

    # 4. ファイル保存とコンソール出力
    log_content = "\n".join(log_lines)
    
    # コンソールに表示
    print("\n" + log_content + "\n")

    # ファイルに保存
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write(log_content)
    
    print(f"ログを保存しました: {LOG_PATH}")

if __name__ == "__main__":
    main()