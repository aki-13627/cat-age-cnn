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

# ==========================================
# ★設定: 学習済みモデルと対象画像のパス
# ==========================================
# 学習したSE-ResNet (Ordinal Regression) のパス
MODEL_PATH = 'outputs/checkpoints/se_resnet50_ordinal_regression_20260105-204440.pth' 

# 予測したい画像のパス
IMAGE_PATH = "data/any_to_predict/スクリーンショット 2026-01-06 13.00.06.png"

# 出力画像の保存先
OUTPUT_FILENAME = "outputs/prediction_with_heatmap.png"

# デバイス設定
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Ordinal Regression設定
NUM_CLASSES = 23
NUM_OUTPUTS = NUM_CLASSES - 1  # 22

# ==========================================
# Grad-CAM用のクラス定義
# ==========================================
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
        
        # 3. Backward Pass (注目箇所を計算)
        self.model.zero_grad()
        
        # どの出力（閾値）に着目して逆伝播するか？
        # 基本的に「予測された年齢の直前の閾値(Yesと答えた最後の質問)」に着目します
        # 例: 5歳と予測 -> index 4 (4歳より大きい?) の反応を見る
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
    # Grad-CAM用にリサイズのみ行った画像(表示用)と、Tensor(推論用)を用意
    raw_image = Image.open(image_path).convert("RGB")
    
    # 表示用リサイズ
    resize_transform = transforms.Compose([
        transforms.Resize((224, 224)),
    ])
    vis_image = resize_transform(raw_image)
    
    # 推論用変換
    normalize_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    input_tensor = normalize_transform(raw_image).unsqueeze(0).to(device)
    return vis_image, input_tensor

# ==========================================
# ヒートマップ合成と表示
# ==========================================
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
    # 1. モデルロード
    model = load_model()

    # 2. Grad-CAMの準備 (最後の畳み込み層をターゲットにする)
    # SE-ResNet50の場合: layer4の最後のボトネック層のconv3あたりを見るのが一般的
    # target_layer = model.layer4[-1].conv3 # もしくは model.layer4[-1] 全体
    target_layer = model.layer4[-1]
    
    grad_cam = GradCAM(model, target_layer)

    # 3. 画像読み込み
    vis_image, input_tensor = preprocess_image(IMAGE_PATH)

    # 4. 実行
    mask, predicted_age = grad_cam(input_tensor)

    print("-" * 30)
    print(f"画像: {IMAGE_PATH}")
    print(f"予測年齢: {predicted_age} 歳")
    print("-" * 30)

    # 5. 画像生成
    overlay, heatmap_img = show_cam_on_image(vis_image, mask)

    # 6. 表示と保存
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