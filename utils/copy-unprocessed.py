import os
import shutil

raw_dir = "data/new_raw"
processed_dir = "data/processed"

for file_name in os.listdir(raw_dir):
    if not (file_name.endswith(".jpg") or file_name.endswith(".png")):
        continue

    base_name = os.path.splitext(file_name)[0]  # 例: "00017_00_02_00"
    pattern_prefix = f"{base_name}_face_"

    # processed フォルダに _face_ 付きのファイルが存在するか確認
    face_exists = any(fname.startswith(pattern_prefix) for fname in os.listdir(processed_dir))

    if not face_exists:
        src_path = os.path.join(raw_dir, file_name)
        dst_path = os.path.join(processed_dir, f"{base_name}_face_0.jpg")
        shutil.copy(src_path, dst_path)
        print(f"{file_name} に対応する _face_ ファイルがないためコピーしました: {dst_path}")
