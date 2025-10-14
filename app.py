import os
import io
import torch
import streamlit as st
import numpy as np
from PIL import Image
import torch.nn.functional as F

# Si vas a cargar desde URL privada, define MODEL_URL en Secrets
# (Streamlit Cloud: App -> Settings -> Secrets)
# Ejemplo:
# [secrets]
# MODEL_URL = "https://github.com/tuuser/turepo/releases/download/v1.0/best_valloss.pth"
import urllib.request

st.set_page_config(page_title="CrackFormer-II | Demo", page_icon="🧱", layout="centered")
st.title("CrackFormer-II — Segmentation Demo")
st.caption("Sube una imagen y obtén el mapa de grietas (probabilidad + máscara).")

# ---------- 1) Arquitectura ----------
from crackformerII import crackformer  # tu archivo local con la definición del modelo

MODEL_LOCAL_PATH = "models/best_valloss.pth"
MODEL_URL = st.secrets.get("MODEL_URL", None)  # opcional: si lo cargas desde release/url
DEVICE = torch.device("cpu")

@st.cache_resource(show_spinner=True)
def load_model():
    os.makedirs(os.path.dirname(MODEL_LOCAL_PATH), exist_ok=True)

    # Descarga si no existe localmente y hay URL
    if not os.path.exists(MODEL_LOCAL_PATH) and MODEL_URL:
        st.info("Descargando pesos del modelo…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_LOCAL_PATH)

    model = crackformer()
    # Carga robusta del checkpoint (como el tuyo de entrenamiento)
    ckpt = torch.load(MODEL_LOCAL_PATH, map_location=DEVICE)
    # Tu guardado usa keys: "model_state", "optimizer_state", etc.
    state = ckpt.get("model_state", ckpt)
    # A veces vienen con prefijos; intentamos carga laxa:
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        st.warning(f"State dict con llaves faltantes/extra (strict=False):\n"
                   f"- faltantes: {missing}\n- extra: {unexpected}")

    model.to(DEVICE).eval()
    return model

model = load_model()

# ---------- 2) Utilidades ----------
def preprocess_pil(img: Image.Image, size=512):
    """RGB -> tensor [1,3,H,W], normalizado (x-0.5)/0.5 como en tu training."""
    img = img.convert("RGB")
    orig_size = img.size  # (W,H)
    img_resized = img.resize((size, size), Image.BILINEAR)
    arr = np.asarray(img_resized).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    x = torch.from_numpy(arr).permute(2,0,1).unsqueeze(0)  # [1,3,H,W]
    return x, orig_size, img

def postprocess_logits(logits, orig_size, threshold=0.5):
    """logits [1,1,h,w] -> prob [H,W], mask uint8 [H,W] a tamaño original."""
    # Probabilidades
    probs = torch.sigmoid(logits)  # [1,1,h,w]
    # Resize back to original size
    H, W = orig_size[1], orig_size[0]
    probs_up = F.interpolate(probs, size=(H, W), mode="bilinear", align_corners=False)
    prob = probs_up[0,0].detach().cpu().numpy()
    mask = (prob >= float(threshold)).astype(np.uint8) * 255
    return prob, mask

def overlay_mask_on_image(img_pil: Image.Image, mask_uint8: np.ndarray, alpha=0.4):
    """Devuelve una imagen PIL con overlay de la máscara (rojo semitransparente)."""
    import cv2
    img = np.array(img_pil.convert("RGB"))
    h, w, _ = img.shape
    if mask_uint8.shape[:2] != (h, w):
        mask_uint8 = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST)
    # color rojo para la máscara
    overlay = img.copy()
    overlay[mask_uint8 > 0] = [255, 0, 0]
    out = (img * (1 - alpha) + overlay * alpha).astype(np.uint8)
    return Image.fromarray(out)

# ---------- 3) UI ----------
uploaded = st.file_uploader("Sube una imagen (jpg/png)", type=["jpg", "jpeg", "png"])
col_t, col_s = st.columns([3,1])
with col_s:
    size = st.selectbox("Tamaño de entrada", [512, 384, 256], index=0)
    thr = st.slider("Umbral máscara", 0.0, 1.0, 0.5, 0.01)
    show_prob = st.checkbox("Mostrar probabilidad", value=True)
    show_mask = st.checkbox("Mostrar máscara", value=True)
    show_overlay = st.checkbox("Mostrar overlay", value=True)

if uploaded is not None:
    img = Image.open(uploaded)
    x, orig_size, img_rgb = preprocess_pil(img, size=int(size))

    with st.spinner("Inferencia…"):
        with torch.no_grad():
            outs = model(x.to(DEVICE))
            # Tu modelo devuelve lista (deep supervision). Tomamos el último mapa:
            if isinstance(outs, (list, tuple)):
                logits = outs[-1]
            else:
                logits = outs
            # Asegurar canal 1
            if logits.shape[1] != 1:
                logits = logits[:, :1, ...]

            prob, mask = postprocess_logits(logits, orig_size, threshold=thr)

    st.subheader("Resultados")
    st.image(img_rgb, caption="Entrada", use_container_width=True)

    cols = st.columns(3)
    idx = 0
    if show_prob:
        # Mostrar probabilidad como imagen en escala de grises
        prob_img = Image.fromarray((prob * 255).astype(np.uint8))
        cols[idx].image(prob_img, caption="Probabilidad", use_container_width=True)
        idx += 1
    if show_mask:
        mask_img = Image.fromarray(mask)
        cols[idx].image(mask_img, caption=f"Máscara (thr={thr:.2f})", use_container_width=True)
        idx += 1
    if show_overlay:
        overlay = overlay_mask_on_image(img_rgb, mask)
        cols[idx].image(overlay, caption="Overlay", use_container_width=True)
