import os
import cv2
from PIL import Image
import numpy as np

input_dir = "/Users/akihiro/cat-age-cnn/data/processed"
output_dir = "/Users/akihiro/cat-age-cnn/data/processed-for-cnn"
image_size = (224, 224)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

def preprocess_image(image):
    resized_img = cv2.resize(image, image_size)

    return resized_img / 255.0

for img_name in os.listdir(input_dir):
    if img_name.endswith(".jpg") or img_name.endswith(".png"):
        img_path = os.path.join(input_dir, img_name)

        img = cv2.imread(img_path)
        if img is None:
            print(f"画像の読み込みに失敗しました: {img_path}")
            continue

        processed_img = preprocess_image(img)

        output_path = os.path.join(output_dir, img_name)

        if success := cv2.imwrite(
            output_path, (processed_img * 255).astype(np.uint8)
        ):
            print(f"画像を保存しました: {output_path}")
        else:
            print(f"画像の保存に失敗しました: {output_path}")

print("画像の前処理と保存が完了しました！")
