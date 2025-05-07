import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import numpy as np
from models.se_resnet import se_resnet18
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import time

# === モデルロード ===
model = se_resnet18(num_classes=23)
model.load_state_dict(
    torch.load("outputs/checkpoints/resnet18_cat_age_se_20250424-182328.pth")
)
model = model.to("mps")
model.eval()

# === 入力画像の前処理 ===
image_path = "data/any_to_predict/スクリーンショット 2025-04-23 17.33.29.png"
img = Image.open(image_path).convert("RGB")

transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)
input_tensor = transform(img).unsqueeze(0)

# === Grad-CAM のターゲット層を指定（layer4がよく使われる） ===
target_layers = [model.layer4[-1]]

cam = GradCAM(model=model, target_layers=target_layers)
with torch.no_grad():
    outputs = model(input_tensor.to("mps"))
    _, predicted_class = torch.max(outputs, 1)
    print(f"予測された年齢（クラス）: {predicted_class.item()} 歳")
targets = [ClassifierOutputTarget(0)]  # 任意クラスの出力に対するCAM（例: class=0）

# === CAM生成 ===
grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

# === 元画像 + CAM 重ねる ===
rgb_img = np.array(img.resize((224, 224))) / 255.0
visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

timestamp = time.strftime("%Y%m%d-%H%M%S")
filename = f"outputs/images/attention_masked_image_{timestamp}.png"
# === 保存・表示 ===
plt.imshow(visualization)
plt.axis("off")
plt.savefig(filename)
plt.show()
