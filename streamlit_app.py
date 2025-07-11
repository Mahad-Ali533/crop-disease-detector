# streamlit_app.py
import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# 1.  Paths
# ─────────────────────────────────────────────────────────────
BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.pt"   # <- keep model here

# ─────────────────────────────────────────────────────────────
# 2.  Class list  (⚠️ Must match training time)
# ─────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_YellowLeaf_Curl_Virus",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_healthy",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "PlantVillage",
]
NUM_CLASSES = len(CLASS_NAMES)

# ─────────────────────────────────────────────────────────────
# 3.  Load model once (cached)
# ─────────────────────────────────────────────────────────────
st.cache_resource(show_spinner="Loading model …")
def load_model():
    model = models.mobilenet_v2(pretrained=False)
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model()   # load right away so the first user hit is fast

# ─────────────────────────────────────────────────────────────
# 4.  Image pre‑processing
# ─────────────────────────────────────────────────────────────
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# ─────────────────────────────────────────────────────────────
# 5.  Streamlit UI
# ─────────────────────────────────────────────────────────────
st.title("🌿 Crop_Disease Detector (MobileNet_V2)")
st.markdown(
    "Upload a plant_leaf photo and get an instant prediction. "
    "Make sure the leaf is clearly visible."
)

uploaded = st.file_uploader("📤  Choose an image …", type=["jpg", "jpeg", "png"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_column_width=True)

    # Inference
    tensor = preprocess(image).unsqueeze(0)           # [1, 3, 224, 224]
    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, 1)
        conf, idx = torch.max(probs, 1)

    st.success(f"**Prediction:** {CLASS_NAMES[idx]}  \n**Confidence:** {conf.item()*100:.1f}%")

