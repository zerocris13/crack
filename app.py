import os
import torch
import streamlit as st
import numpy as np
from PIL import Image
import torch.nn.functional as F
import urllib.request

st.set_page_config(page_title="CrackFormer-II | Demo", page_icon="🧱", layout="centered")
st.title("Segmentation Demo")
#st.caption("Sube una imagen y visualiza entrada, máscara (umbral) y overlay en una fila.")

# ---------- 1) Modelo ----------
from crackformerII import crackformer  
MODEL_LOCAL_PATH = "models/best_valloss.pth"
MODEL_URL = st.secrets.get("MODEL_URL", None)  
DEVICE = torch.device("cpu")

@st.cache_resource(show_spinner=True)
def load_model():
    os.makedirs(os.path.dirname(MODEL_LOCAL_PATH), exist_ok=True)
    if not os.path.exists(MODEL_LOCAL_PATH) and MODEL_URL:
        st.info("Descargando pesos del modelo…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_LOCAL_PATH)

    model = crackformer()
    ckpt = torch.load(MODEL_LOCAL_PATH, map_location=DEVICE)
    state = ckpt.get("model_state", ckpt)  # tu checkpoint guarda 'model_state'
    model.load_state_dict(state, strict=False)
    model.to(DEVICE).eval()
    return model

model = load_model()

# ---------- 2) Utilidades ----------
def preprocess_pil(img: Image.Image, size=512):
    img = img.convert("RGB")
    orig_size = img.size  # (W,H)
    img_resized = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img_resized).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    x = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)  # [1,3,H,W]
    return x, orig_size, img

def postprocess_logits(logits, orig_size, threshold=0.5):
    probs = torch.sigmoid(logits)  # [1,1,h,w]
    H, W = orig_size[1], orig_size[0]
    probs_up = F.interpolate(probs, size=(H, W), mode="bilinear", align_corners=False)
    prob = probs_up[0,0].detach().cpu().numpy()
    mask = (prob >= float(threshold)).astype(np.uint8) * 255
    return mask

def overlay_mask_on_image(img_pil: Image.Image, mask_uint8: np.ndarray, alpha=0.4):
    import cv2
    img = np.array(img_pil.convert("RGB"))
    h, w, _ = img.shape
    if mask_uint8.shape[:2] != (h, w):
        mask_uint8 = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST)
    overlay = img.copy()
    overlay[mask_uint8 > 0] = [255, 0, 0]  # rojo
    out = (img * (1 - alpha) + overlay * alpha).astype(np.uint8)
    return Image.fromarray(out)

# ---------- 3) UI ----------
uploaded = st.file_uploader("Sube una imagen (jpg/png)", type=["jpg", "jpeg", "png"])
col_controls, _ = st.columns([1,3])
with col_controls:
    size = st.selectbox("Tamaño de entrada", [512, 384, 256], index=0)
    thr = st.slider("Umbral máscara", 0.0, 1.0, 0.5, 0.01)

if uploaded is not None:
    img = Image.open(uploaded)
    x, orig_size, img_rgb = preprocess_pil(img, size=int(size))

    with st.spinner("Inferencia…"):
        with torch.no_grad():
            outs = model(x.to(DEVICE))
            logits = outs[-1] if isinstance(outs, (list, tuple)) else outs
            if logits.shape[1] != 1:
                logits = logits[:, :1, ...]
            mask = postprocess_logits(logits, orig_size, threshold=thr)

    # --------- 4) Visualización en una fila ----------
    c1, c2, c3 = st.columns(3)
    c1.image(img_rgb, caption="Entrada", use_container_width=True)
    c2.image(Image.fromarray(mask), caption=f"Máscara (thr={thr:.2f})", use_container_width=True)
    c3.image(overlay_mask_on_image(img_rgb, mask), caption="Overlay", use_container_width=True)
