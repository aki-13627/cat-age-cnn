import sys
# パスはご自身の環境に合わせて調整してください
sys.path.append('/Users/akihiro/cat-age-cnn')

import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
from models.se_resnet import se_resnet50





MODEL_PATH = 'outputs/checkpoints/se_resnet50_ordinal_regression_20260106-174333.pth' 


IMAGE_PATH = "data/any_to_predict/2026-01-06 16.10.15_15.png"


OUTPUT_FILENAME = "outputs/prediction_with_heatmap.png"


device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


NUM_CLASSES = 23
NUM_OUTPUTS = NUM_CLASSES - 1  # 22



class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # フックの登録
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __call__(self, x, class_idx=None):
        # 1. Forward Pass
        output = self.model(x)

        # 2. Ordinal Regressionの推論
        probs = torch.sigmoid(output)
        predicted_age = torch.sum(probs > 0.5).item()
        
        self.model.zero_grad()
        if predicted_age > 0:
            target_index = int(predicted_age - 1)
        else:
            target_index = 0 # 0歳の場合は最初の閾値を見る

        # 特定の出力に対してBackwardを実行
        one_hot = torch.zeros_like(output)
        one_hot[0][target_index] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        # 4. Grad-CAMの計算
        # Global Average Pooling of Gradients
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        
        # 重み付けして足し合わせる
        activation = self.activations[0].detach().cpu().numpy()
        for i in range(activation.shape[0]):
            activation[i, :, :] *= pooled_gradients[i].item()

        heatmap = np.mean(activation, axis=0)
        
        # ReLU (マイナスの寄与は無視)
        heatmap = np.maximum(heatmap, 0)
        
        # 正規化 (0~1)
        if np.max(heatmap) != 0:
            heatmap /= np.max(heatmap)
            
        return heatmap, predicted_age

# ==========================================
# モデルと画像の準備
# ==========================================
def load_model():
    print(f"Loading model from: {MODEL_PATH}")
    model = se_resnet50(num_classes=1000)
    num_ftrs = 2048 # ResNet50固定
    
    model.fc = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(num_ftrs, NUM_OUTPUTS)
    )
    
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=False))
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

    model = model.to(device)
    model.eval()
    return model

def preprocess_image(image_path):
    
    raw_image = Image.open(image_path).convert("RGB")
    
    
    resize_transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])
    vis_image = resize_transform(raw_image)
    
    
    normalize_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = normalize_transform(raw_image).unsqueeze(0).to(device)
    return vis_image, input_tensor

def show_cam_on_image(img, mask):
    # 画像をnumpy配列に変換
    img = np.array(img)
    img = img.astype(np.float32) / 255
    
    # ヒートマップを画像サイズにリサイズ
    heatmap = cv2.resize(mask, (img.shape[1], img.shape[0]))
    
    # カラーマップ適用 (青->赤)
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    # 合成
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam), np.uint8(255 * heatmap)

# ==========================================
# メイン処理
# ==========================================
if __name__ == "__main__":

    model = load_model()

    target_layer = model.layer4[-1]
    
    grad_cam = GradCAM(model, target_layer)


    vis_image, input_tensor = preprocess_image(IMAGE_PATH)


    mask, predicted_age = grad_cam(input_tensor)

    print("-" * 30)
    print(f"画像: {IMAGE_PATH}")
    print(f"予測年齢: {predicted_age} 歳")
    print("-" * 30)


    overlay, heatmap_img = show_cam_on_image(vis_image, mask)


    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(vis_image)
    plt.title(f"Original (Pred: {predicted_age})")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(heatmap_img)
    plt.title("Attention Heatmap")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Overlay")
    plt.axis('off')

    plt.tight_layout()
    plt.savefig(OUTPUT_FILENAME)
    print(f"結果画像を保存しました: {OUTPUT_FILENAME}")
    plt.show()