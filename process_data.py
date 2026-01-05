import os

TARGET_DIR = "data/processed"
DRY_RUN = False  # Trueでプレビュー、Falseで実行

def main():
    if not os.path.exists(TARGET_DIR):
        print(f"Error: Directory '{TARGET_DIR}' not found.")
        return

    print(f"--- Checking extensions in {TARGET_DIR} ---")
    
    count = 0
    files = os.listdir(TARGET_DIR)
    
    for filename in files:
        # 隠しファイルは無視
        if filename.startswith('.'):
            continue

        root, ext = os.path.splitext(filename)
        
        # 拡張子が .jpg 以外（.JPG, .jpeg など）の場合のみ処理
        if ext != '.jpg' and ext.lower() in ['.jpg', '.jpeg']:
            new_name = f"{root}.jpg"
            
            old_path = os.path.join(TARGET_DIR, filename)
            new_path = os.path.join(TARGET_DIR, new_name)
            
            if DRY_RUN:
                print(f"[Preview] {filename} -> {new_name}")
            else:
                # macOSなどで大文字小文字だけの変更がうまく認識されない場合の対策として
                # 一度別名を経由させるのが確実ですが、os.renameで基本はいけます。
                try:
                    os.rename(old_path, new_path)
                    print(f"[Renamed] {filename} -> {new_name}")
                except OSError as e:
                    print(f"[Error] Failed to rename {filename}: {e}")
            
            count += 1

    if count == 0:
        print("変更が必要なファイルはありませんでした。（全てすでに .jpg です）")
    else:
        if DRY_RUN:
            print(f"\n--- {count} files will be renamed ---")
            print("Set DRY_RUN = False to execute.")
        else:
            print(f"\n--- {count} files renamed ---")

if __name__ == "__main__":
    main()