import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models.vision_transformer import vit_b_16
from PIL import Image
import numpy as np
import cv2
import io
import zipfile

class ViTHybridModel(nn.Module):
    def __init__(self, num_classes=4, lstm_hidden_size=128):
        super(ViTHybridModel, self).__init__()
        self.vit = vit_b_16(pretrained=True)
        self.vit.heads = nn.Identity()
        self.lstm = nn.LSTM(64, lstm_hidden_size, 2, batch_first=True, dropout=0.3, bidirectional=True)
        self.fog_encoder = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 64))
        self.fusion = nn.Sequential(
            nn.Linear(768 + 64 + 2 * lstm_hidden_size, 256),
            nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2)
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, images, lidar, fog_feats):
        vit_feat = self.vit(images)
        lstm_out, _ = self.lstm(lidar)
        lstm_feat = lstm_out[:, -1, :]
        fog_enc = self.fog_encoder(fog_feats)
        x = torch.cat([vit_feat, lstm_feat, fog_enc], dim=1)
        x = self.fusion(x)
        return self.classifier(x)

def estimate_fog_density(img):
    img_np = np.array(img)
    if img_np.dtype != np.uint8:
        img_np = (img_np * 255).clip(0, 255).astype(np.uint8)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    return np.var(gray)

def overlay_prediction(image, label, confidence):
    img_np = np.array(image)
    overlay = img_np.copy()
    text = f"{label} ({confidence:.2f}%)"
    cv2.putText(overlay, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return Image.fromarray(overlay)

def get_dark_channel(img, patch_size):
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    return cv2.erode(min_channel, kernel)

def estimate_atmospheric_light(img, dark_channel, top_percent=0.001):
    img_size = dark_channel.size
    num_pixels = int(max(1, img_size * top_percent))
    flat_dark = dark_channel.flatten()
    indices = np.argsort(flat_dark)[-num_pixels:]
    h, w = dark_channel.shape
    coords = np.unravel_index(indices, (h, w))
    a_max, a_idx = 0, 0
    for i in range(len(indices)):
        y, x = coords[0][i], coords[1][i]
        pixel_intensity = np.mean(img[y, x, :])
        if pixel_intensity > a_max:
            a_max = pixel_intensity
            a_idx = i
    y, x = coords[0][a_idx], coords[1][a_idx]
    return img[y, x, :]

def estimate_transmission(img, A, omega, patch_size):
    normalized_img = np.empty(img.shape, img.dtype)
    for c in range(3):
        normalized_img[:, :, c] = img[:, :, c] / max(A[c], 1e-3)
    dark_channel = get_dark_channel(normalized_img, patch_size)
    return 1 - omega * dark_channel

def guided_filter(guide, src, radius, eps):
    h, w = guide.shape[:2]
    N = cv2.boxFilter(np.ones((h, w)), -1, (radius, radius))
    mean_I = cv2.boxFilter(guide, -1, (radius, radius)) / N
    mean_p = cv2.boxFilter(src, -1, (radius, radius)) / N
    mean_Ip = cv2.boxFilter(guide * src, -1, (radius, radius)) / N
    cov_Ip = mean_Ip - mean_I * mean_p
    mean_II = cv2.boxFilter(guide * guide, -1, (radius, radius)) / N
    var_I = mean_II - mean_I * mean_I
    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I
    mean_a = cv2.boxFilter(a, -1, (radius, radius)) / N
    mean_b = cv2.boxFilter(b, -1, (radius, radius)) / N
    return mean_a * guide + mean_b

def dehaze_image(pil_img, patch_size=9, omega=0.85, min_transmission=0.15, radius=20, eps=1e-2, top_percent=0.001):
    img_np = np.array(pil_img).astype(np.float32) / 255.0
    dark_channel = get_dark_channel(img_np, patch_size)
    A = estimate_atmospheric_light(img_np, dark_channel, top_percent)
    transmission = estimate_transmission(img_np, A, omega, patch_size)
    gray_img = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    refined_trans = guided_filter(gray_img, transmission, radius, eps)
    refined_trans = np.maximum(refined_trans, min_transmission)
    dehazed = np.zeros_like(img_np)
    for c in range(3):
        dehazed[:, :, c] = (img_np[:, :, c] - A[c]) / refined_trans + A[c]
    dehazed = np.clip(dehazed, 0.0, 1.0)
    dehazed = (dehazed * 255).astype(np.uint8)
    return Image.fromarray(dehazed)

st.set_page_config(page_title="ViT Fog Detection & Defogging", layout="wide")
st.title("🌫️ ViT-Based Fog Detection and Defogging")
st.subheader("Upload images for fog classification and defogging")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fog_classes = {
    0: 'Clear',
    1: 'Homogeneous Fog',
    2: 'Inhomogeneous Fog',
    3: 'Sky Fog'
}

model = ViTHybridModel(num_classes=4)
model.load_state_dict(torch.load("models/best_vit_model.pth", map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

uploaded_files = st.file_uploader("Upload one or more images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_files:
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED) as zipf:
        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert('RGB')
            input_tensor = transform(image).unsqueeze(0).to(device)
            fog_val = estimate_fog_density(image)
            fog_tensor = torch.tensor([[fog_val, 0, 0, 0]], dtype=torch.float32).to(device)
            lidar_dummy = torch.zeros((1, 10, 64), dtype=torch.float32).to(device)

            with torch.no_grad():
                output = model(input_tensor, lidar_dummy, fog_tensor)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                predicted_label = fog_classes[predicted_class.item()]
                confidence = confidence.item() * 100

            defogged_img = dehaze_image(image)
            defog_tensor = transform(defogged_img).unsqueeze(0).to(device)

            with torch.no_grad():
                out_defog = model(defog_tensor, lidar_dummy, fog_tensor)
                prob_defog = torch.softmax(out_defog, dim=1)
                conf_defog, pred_defog = torch.max(prob_defog, 1)
                label_defog = fog_classes[pred_defog.item()]
                conf_defog = conf_defog.item() * 100

            image_annotated = overlay_prediction(image, predicted_label, confidence)
            defogged_annotated = overlay_prediction(defogged_img, label_defog, conf_defog)

            cols = st.columns(2)
            with cols[0]:
                st.image(image_annotated, caption=f"Original Prediction: {predicted_label} ({confidence:.2f}%)", use_container_width=True)
            with cols[1]:
                st.image(defogged_annotated, caption=f"Defogged Prediction: {label_defog} ({conf_defog:.2f}%)", use_container_width=True)

            buf = io.BytesIO()
            defogged_img.save(buf, format="PNG")
            zipf.writestr(f"defogged_{uploaded_file.name}", buf.getvalue())

    st.download_button(
        label="📦 Download All Defogged Images as ZIP",
        data=zip_buffer.getvalue(),
        file_name="defogged_vit_images.zip",
        mime="application/zip"
    )
