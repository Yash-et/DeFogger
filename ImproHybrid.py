import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms, models
import io
import zipfile

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

class ImprovedHybridCNNLSTM(nn.Module):
    def __init__(self, num_classes=4, lstm_hidden_size=128, use_pretrained=True):
        super(ImprovedHybridCNNLSTM, self).__init__()
        self.cnn = models.resnet50(pretrained=use_pretrained)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])
        self.lstm = nn.LSTM(input_size=64, hidden_size=lstm_hidden_size, num_layers=2, batch_first=True, dropout=0.2, bidirectional=True)
        self.fog_feature_encoder = nn.Sequential(nn.Linear(4, 32), nn.ReLU(), nn.Linear(32, 64), nn.ReLU())
        self.fusion = nn.Sequential(nn.Linear(2048 + lstm_hidden_size*2 + 64, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.2))
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, images, lidar_sequences, fog_features):
        batch_size = images.size(0)
        cnn_features = self.cnn(images).view(batch_size, -1)
        lstm_out, _ = self.lstm(lidar_sequences)
        lstm_features = lstm_out[:, -1, :]
        encoded_fog_features = self.fog_feature_encoder(fog_features)
        combined_features = torch.cat((cnn_features, lstm_features, encoded_fog_features), dim=1)
        fused_features = self.fusion(combined_features)
        output = self.classifier(fused_features)
        return output

def overlay_text(img, text):
    img_copy = img.copy()
    draw = ImageDraw.Draw(img_copy)
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except:
        font = ImageFont.load_default()
    draw.text((10, 10), text, fill="yellow", font=font)
    return img_copy

st.set_page_config(page_title="Fog Detection + Defogging", layout="wide")
st.title("🌫️ Fog Type Detection and Image Defogging 🚗")
st.subheader("Upload multiple images to detect fog type and defog them!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fog_classes = {
    0: 'Clear',
    1: 'Homogeneous Fog',
    2: 'Inhomogeneous Fog',
    3: 'Sky Fog'
}

model = ImprovedHybridCNNLSTM(num_classes=4, lstm_hidden_size=128)
model.load_state_dict(torch.load('models/best_fog_detection_model.pth', map_location=device))
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
            dummy_lidar = torch.zeros((1, 10, 64), dtype=torch.float32).to(device)
            dummy_fog_feature = torch.zeros((1, 4), dtype=torch.float32).to(device)
            
            with torch.no_grad():
                output = model(input_tensor, dummy_lidar, dummy_fog_feature)
                probabilities = torch.softmax(output, dim=1)
                confidence, predicted_class = torch.max(probabilities, 1)
                predicted_label = fog_classes[predicted_class.item()]
                confidence = confidence.item() * 100
            
            defogged_img = dehaze_image(image)
            label_text = f"{predicted_label} ({confidence:.2f}%)"
            image_with_text = overlay_text(image, label_text)
            defogged_with_text = overlay_text(defogged_img, label_text)

            cols = st.columns(2)
            with cols[0]:
                st.image(image_with_text, caption="Original with Confidence", use_container_width=True)
            with cols[1]:
                st.image(defogged_with_text, caption="Defogged with Confidence", use_container_width=True)

            buf = io.BytesIO()
            defogged_img.save(buf, format="PNG")
            zipf.writestr(f"defogged_{uploaded_file.name}", buf.getvalue())

    st.download_button(
        label="Download All Defogged Images as ZIP",
        data=zip_buffer.getvalue(),
        file_name="defogged_images.zip",
        mime="application/zip"
    )
