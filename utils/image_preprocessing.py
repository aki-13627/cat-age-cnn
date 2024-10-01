import torch
from torchvision import models, transforms
import cv2
import os
import numpy as np
from PIL import Image

input_dir = "/Users/akihiro/cat-age-cnn/data/raw"
output_dir = "/Users/akihiro/cat-age-cnn/data/processed"

model = models.detection.ssd300_vgg16(pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.ToTensor(),
])

for img_name in os.listdir(input_dir):
    if img_name.endswith(".jpg") or img_name.endswith(".png"):
        img_path = os.path.join(input_dir, img_name)

        img = Image.open(img_path)
        img_tensor = preprocess(img).unsqueeze(0)
        
        with torch.no_grad():
            predictions = model(img_tensor)
            
        detected_faces = predictions[0]['boxes']
        scores = predictions[0]['scores']

        for i, (box, score) in enumerate(zip(detected_faces, scores)):
            if score > 0.5:
                x1, y1, x2, y2 = map(int, box)
                print(f"顔が検出されました: {img_name} スコア: {score.item()}")

                img_cv = cv2.imread(img_path)
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0, 255, 0), 2)

                face_img = img_cv[y1:y2, x1:x2]
                face_height, face_width = face_img.shape[:2]
                if face_height * face_width < 2000000:
                    print(f"出力画像が小さすぎます: {img_name}_face_{i} ({face_height * face_width} pixels), スキップします")
                    continue
                if face_height * face_width > 9000000:
                    print(f"出力画像が大きすぎます: {img_name}_face_{i} ({face_height * face_width} pixels), スキップします")
                    continue
                output_path = os.path.join(output_dir, f"{os.path.splitext(img_name)[0]}_face_{i}.jpg")
                cv2.imwrite(output_path, face_img)
                print(f"Saved: {output_path}")

print("顔検出と保存が完了しました！")
