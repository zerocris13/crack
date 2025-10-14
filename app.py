import os
import urllib.request
import numpy as np
import torch
import torch.nn.functional as F
import streamlit as st
from PIL import Image

# ------------------ Config UI ------------------
st.set_page_config(page_title="CrackFormer-II | Demo", page_icon="🧱", layout="centered")
st.title("Segmentation Demo")

# ------------------ Modelo ------------------
from crackformerII import crackformer  # Asegúrate de tener crackformerII.py junto a este archivo

MODEL_LOCAL_PATH = "models/best_valloss.pth"
MODEL_URL = st.secrets.get("MODEL_URL", None)  # opcional: URL de release/host para descargar pesos
DEVICE = torch.device("cpu")


@st.cache_resource(show_spinner=True)
def load_model():
    os.makedirs(os.path.dirname(MODEL_LOCAL_PATH), exist_ok=True)

    def is_suspect(path, min_bytes=1024 * 1024):
        # considera sospechoso si no existe o pesa < 1 MB
        return (not os.path.exists(path)) or (os.path.getsize(path) < min_bytes)

    # Descarga si el archivo no está o es sospechoso y tienes MODEL_URL
    if is_suspect(MODEL_LOCAL_PATH) and MODEL_URL:
        st.info("Descargando pesos del modelo…")
        urllib.request.urlretrieve(MODEL_URL, MODEL_LOCAL_PATH)

    model = crackformer()

    def try_load():
        ckpt = torch.load(MODEL_LOCAL_PATH, map_location=DEVICE)
        state = ckpt.get("model_state", ckpt)  # tu entrenamiento guardó 'model_state'
        model.load_state_dict(state, strict=False)

    try:
        try_load()
    except Exception as e:
        # Reintento descargando si hay URL
        if MODEL_URL:
            st.warning(f"No se pudieron leer los pesos ({type(e).__name__}). Reintentando con descarga…")
            urllib.request.urlretrieve(MODEL_URL, MODEL_LOCAL_PATH)
            try_load()
        else:
            sz = os.path.getsize(MODEL_LOCAL_PATH) if os.path.exists(MODEL_LOCAL_PATH) else 0
            raise RuntimeError(
                f"No se pudo cargar '{MODEL_LOCAL_PATH}'. "
                f"¿Archivo corrupto o subida incompleta? Tamaño actual: {sz} bytes. "
                f"Vuelve a subirlo o configura MODEL_URL en Secrets."
            ) from e

    model.to(DEVICE).eval()
    return model


model = load_model()

# ------------------ Utilidades ------------------
def preprocess_pil(img: Image.Image):
    """Convierte la imagen a tensor manteniendo su tamaño original."""
    img = img.convert("RGB")
    orig_size = img.size  # (W, H)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)  # [1,3,H,W]
    return x, orig_size, img


def postprocess_logits(logits, orig_size, threshold=0.5):
    """De logits -> máscara binaria uint8 del tamaño original."""
    probs = torch.sigmoid(logits)  # [1,1,h,w]
    H, W = orig_size[1], orig_size[0]
    probs_up = F.interpolate(probs, size=(H, W), mode="bilinear", align_corners=False)
    prob = probs_up[0, 0].detach().cpu().numpy()
    mask = (prob >= float(threshold)).astype(np.uint8) * 255
    return mask


def overlay_mask_on_image(img_pil: Image.Image, mask_uint8: np.ndarray, alpha=0.4):
    """Overlay rojo semitransparente de la máscara sobre la imagen."""
    import cv2  # import local para evitar cargarlo si no se usa
    img = np.array(img_pil.convert("RGB"))
    h, w, _ = img.shape
    if mask_uint8.shape[:2] != (h, w):
        mask_uint8 = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST)
    overlay = img.copy()
    overlay[mask_uint8 > 0] = [255, 0, 0]
    out = (img * (1 - alpha) + overlay * alpha).astype(np.uint8)
    return Image.fromarray(out)


# ------------------ UI: carga e inferencia ------------------
uploaded_file = st.file_uploader(
    "Sube una imagen (JPG o PNG)", type=("jpg", "jpeg", "png"), accept_multiple_files=False, key="file"
)
thr = st.slider("Umbral máscara", 0.0, 1.0, 0.5, 0.01)

if not uploaded_file:
    st.info("Sube una imagen para continuar.")
    st.stop()

# Abrimos la imagen de forma segura
try:
    img = Image.open(uploaded_file).convert("RGB")
except Exception as e:
    st.error(f"No se pudo abrir la imagen: {e}")
    st.stop()

x, orig_size, img_rgb = preprocess_pil(img)

with st.spinner("Inferencia…"):
    with torch.no_grad():
        outs = model(x.to(DEVICE))
        logits = outs[-1] if isinstance(outs, (list, tuple)) else outs
        if logits.shape[1] != 1:
            logits = logits[:, :1, ...]
        mask = postprocess_logits(logits, orig_size, threshold=thr)

# ------------------ Visualización en una fila ------------------
c1, c2, c3 = st.columns(3)
c1.image(img_rgb, caption="Entrada", use_container_width=True)
c2.image(Image.fromarray(mask), caption=f"Máscara (thr={thr:.2f})", use_container_width=True)
c3.image(overlay_mask_on_image(img_rgb, mask), caption="Overlay", use_container_width=True)