"""
app.py
Streamlit demo for the Salient Object Detection model.

Run with:
    streamlit run app.py
"""

import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

from sod_model import build_model

DEFAULT_CHECKPOINT = "checkpoints/improved_best.pth"
DEFAULT_THRESHOLD  = 0.5


@st.cache_resource
def load_model(checkpoint_path: str):
    """
    Loads the model from a checkpoint file.
    @st.cache_resource means this only runs once — the model stays in memory
    between predictions instead of reloading every time someone clicks the button.
    """
    device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt       = torch.load(checkpoint_path, map_location=device)
    cfg        = ckpt.get("config", {})
    model_type = cfg.get("model_type", "baseline")
    img_size   = cfg.get("img_size", 128)

    model = build_model(model_type=model_type, img_size=img_size)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model.to(device), img_size, model_type, device


def run_inference(image_np, model, img_size, device, threshold):
    """
    Takes a raw RGB image (H x W x 3, uint8) and returns:
      - heatmap: the raw saliency probability map, colourised with HOT colormap
      - overlay: the original image with the salient region tinted red
      - elapsed_ms: how long the model took
    """
    orig_h, orig_w = image_np.shape[:2]

    img_r  = cv2.resize(image_np, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    tensor = (
        torch.from_numpy(img_r.astype(np.float32) / 255.0)
        .permute(2, 0, 1).unsqueeze(0).to(device)
    )

    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model(tensor)
    elapsed_ms = (time.perf_counter() - t0) * 1000

    mask     = pred[0, 0].cpu().numpy()
    mask     = cv2.resize(mask, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    mask_bin = (mask > threshold).astype(np.float32)

    mask_u8     = (mask * 255).clip(0, 255).astype(np.uint8)
    heatmap_bgr = cv2.applyColorMap(mask_u8, cv2.COLORMAP_HOT)
    heatmap_rgb = cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB)

    img_f   = image_np.astype(np.float32) / 255.0
    overlay = np.stack([
        np.clip(img_f[..., 0] + mask_bin * 0.45, 0, 1),
        np.clip(img_f[..., 1] * (1 - mask_bin * 0.35), 0, 1),
        np.clip(img_f[..., 2] * (1 - mask_bin * 0.35), 0, 1),
    ], axis=-1)
    overlay_u8 = (overlay * 255).astype(np.uint8)

    return heatmap_rgb, overlay_u8, elapsed_ms


# ── UI ────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Salient Object Detection", layout="wide")
st.title("Salient Object Detection")
st.markdown("Upload an image to detect the most visually dominant object.")
st.divider()

# Settings row — always visible, no sidebar
col_upload, col_ckpt, col_thresh = st.columns([2, 2, 1])

with col_upload:
    uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp"])

with col_ckpt:
    checkpoint_path = st.text_input("Checkpoint path", value=DEFAULT_CHECKPOINT)
    if Path(checkpoint_path).exists():
        _, img_size, model_type, device = load_model(checkpoint_path)
        st.success(f"Model: **{model_type}** | Size: {img_size}px | Device: {device}")
    else:
        st.error("Checkpoint not found. Train the model first.")

with col_thresh:
    threshold = st.slider("Threshold", 0.1, 0.9, value=DEFAULT_THRESHOLD, step=0.05)

st.divider()

# Results
if uploaded is not None and Path(checkpoint_path).exists():
    model, img_size, model_type, device = load_model(checkpoint_path)

    pil_img  = Image.open(uploaded).convert("RGB")
    image_np = np.array(pil_img)

    heatmap, overlay, elapsed_ms = run_inference(image_np, model, img_size, device, threshold)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.subheader("Input Image")
        st.image(image_np, use_container_width=True)
    with col2:
        st.subheader("Saliency Map")
        st.image(heatmap, use_container_width=True)
    with col3:
        st.subheader("Overlay")
        st.image(overlay, use_container_width=True)

    st.info(
        f"Inference time: **{elapsed_ms:.1f} ms** | "
        f"Device: **{device}** | "
        f"Threshold: **{threshold}**"
    )
