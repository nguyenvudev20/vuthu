import os
import io
import json
import time
import math
from typing import List, Dict, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import streamlit as st

# Optional ML: try TensorFlow MobileNetV2; gracefully fall back if not available
TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
except Exception:
    TF_AVAILABLE = False

# Video export
from moviepy.editor import ImageSequenceClip, AudioFileClip

APP_TITLE = "AI Smart Slideshow – Wedding (Streamlit)"
DEFAULT_IMAGE_DIR = "hinhanh"
DEFAULT_MANIFEST = os.path.join(DEFAULT_IMAGE_DIR, "manifest.json")
DEFAULT_AUDIO_CANDIDATES = [
    "audio/xứng đôi cưới thôi nhen.mp3",
    "audio/xung doi cuoi thoi nhen.mp3",
    "audio/xung-doi-cuoi-thoi-nhen.mp3",
]
OUTPUT_VIDEO = "slideshow.mp4"
FPS_EXPORT = 30

EFFECT_OPTIONS = [
    ("auto", "Tự động (ML/Heuristic)"),
    ("fx-fade", "Fade dịu"),
    ("fx-zoom-in", "Zoom in nhẹ"),
    ("fx-zoom-out", "Zoom out nhẹ"),
    ("fx-pan-left", "Pan trái"),
    ("fx-pan-right", "Pan phải"),
    ("fx-tilt", "Tilt nghiêng"),
]

FONT_PATHS = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]

def load_font(size: int = 24) -> ImageFont.FreeTypeFont:
    for p in FONT_PATHS:
        if os.path.exists(p):
            return ImageFont.truetype(p, size=size)
    return ImageFont.load_default()

# ============ ML model (optional) ============
_mobilenet = None

def ensure_model():
    global _mobilenet
    if not TF_AVAILABLE:
        return None
    if _mobilenet is None:
        _mobilenet = MobileNetV2(weights="imagenet")
    return _mobilenet

# ============ Utilities ============

def list_images_from_manifest_or_dir() -> List[Dict]:
    """Return list of dicts: {"path": path, "effect": "auto"}.
    Prefer manifest.json; else glob directory.
    Supports manifest entries as strings or objects {file, effect}.
    """
    images = []
    if os.path.exists(DEFAULT_MANIFEST):
        try:
            data = json.load(open(DEFAULT_MANIFEST, "r", encoding="utf-8"))
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, str):
                        images.append({"path": os.path.join(DEFAULT_IMAGE_DIR, item), "effect": "auto"})
                    elif isinstance(item, dict) and "file" in item:
                        eff = item.get("effect", "auto")
                        images.append({"path": os.path.join(DEFAULT_IMAGE_DIR, item["file"]), "effect": eff})
        except Exception as e:
            st.warning(f"Không đọc được manifest.json: {e}")
    if not images:
        # fallback: list by extension
        if os.path.isdir(DEFAULT_IMAGE_DIR):
            for name in sorted(os.listdir(DEFAULT_IMAGE_DIR)):
                if name.lower().endswith((".jpg", ".jpeg", ".png", ".webp", ".gif")):
                    images.append({"path": os.path.join(DEFAULT_IMAGE_DIR, name), "effect": "auto"})
    return images


def pil_centered_text(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font: ImageFont.ImageFont, fill=(255, 255, 255)):
    w, h = draw.textbbox((0, 0), text, font=font)[2:]
    x, y = xy
    draw.text((x - w // 2, y - h // 2), text, fill=fill, font=font)


def compute_brightness_saturation(img: Image.Image) -> Tuple[float, float]:
    arr = np.asarray(img.convert("RGB").resize((224, 224))) / 255.0
    # luminance
    lum = np.dot(arr[..., :3], [0.2126, 0.7152, 0.0722])
    brightness = float(lum.mean())
    # saturation proxy
    maxc = arr[..., :3].max(axis=2)
    minc = arr[..., :3].min(axis=2)
    sat = np.where(maxc != 0, (maxc - minc) / (maxc + 1e-6), 0)
    saturation = float(sat.mean())
    return brightness, saturation


def classify_top_label(img: Image.Image) -> str:
    model = ensure_model()
    if model is None:
        return ""  # ML not available
    arr = img.convert("RGB").resize((224, 224))
    x = np.array(arr)[None, ...].astype("float32")
    x = preprocess_input(x)
    preds = model.predict(x, verbose=0)
    try:
        label = decode_predictions(preds, top=1)[0][0][1]
    except Exception:
        label = ""
    return label


def choose_effect_auto(img: Image.Image) -> str:
    """Map content to effect using ML if available else heuristic."""
    top = classify_top_label(img) if TF_AVAILABLE else ""
    b, s = compute_brightness_saturation(img)
    if any(k in top.lower() for k in ["person", "bride", "groom", "gown", "suit", "portrait"]):
        return "fx-zoom-in"
    if any(k in top.lower() for k in ["mountain", "valley", "lake", "beach", "ocean", "landscape", "outdoor"]):
        return "fx-pan-left" if (hash(top) % 2 == 0) else "fx-pan-right"
    if s > 0.45 and b > 0.55:
        return "fx-tilt"
    if b < 0.35:
        return "fx-zoom-out"
    return "fx-fade"

# ============ Effect rendering (Ken Burns style) ============

def render_frame(img: Image.Image, effect: str, t: float, duration: float, canvas_size=(1280, 720)) -> Image.Image:
    W, H = canvas_size
    base = Image.new("RGB", (W, H), (15, 17, 24))
    iw, ih = img.size
    # contain fit box
    scale = min(W / iw, H / ih)
    dw, dh = int(iw * scale), int(ih * scale)

    # Ken Burns params
    def lerp(a, b, u):
        return a + (b - a) * u

    u = max(0.0, min(1.0, t / max(0.001, duration)))
    # defaults (no motion)
    sx, sy = 1.0, 1.0
    ox, oy = 0, 0
    angle = 0.0

    if effect == "fx-zoom-in":
        sx = lerp(1.0, 1.12, u)
    elif effect == "fx-zoom-out":
        sx = lerp(1.12, 1.0, u)
    elif effect == "fx-pan-left":
        ox = int(lerp(0.05 * W, -0.05 * W, u))
        sx = 1.06
    elif effect == "fx-pan-right":
        ox = int(lerp(-0.05 * W, 0.05 * W, u))
        sx = 1.06
    elif effect == "fx-tilt":
        angle = lerp(-0.6, 0.6, u)
        sx = 1.05
    # fx-fade handled outside via alpha in preview; for export we just keep static/kenburns

    # scale
    scaled = img.resize((int(dw * sx), int(dh * sx)), Image.LANCZOS)
    sw, sh = scaled.size
    # center coordinates + offset
    x = (W - sw) // 2 + ox
    y = (H - sh) // 2 + oy
    if angle != 0:
        scaled = scaled.rotate(angle, resample=Image.BICUBIC, expand=True)
        sw, sh = scaled.size
        x = (W - sw) // 2 + ox
        y = (H - sh) // 2 + oy
    base.paste(scaled, (x, y))
    return base


def overlay_texts(frame: Image.Image, names_line: str, lyric: str, marquee: bool, t: float) -> Image.Image:
    im = frame.copy()
    draw = ImageDraw.Draw(im)
    # Names (top center)
    if names_line.strip():
        font_names = load_font(48)
        pil_centered_text(draw, (im.width // 2, 60), names_line.replace("\\n", " "), font_names, (255, 255, 255))
    # Lyric (bottom, marquee)
    if lyric.strip():
        font_ly = load_font(22)
        text = lyric
        if marquee:
            # shift right-to-left
            px_per_sec = 120
            shift = int((t * px_per_sec) % (im.width + len(text) * 10))
            x = im.width - shift + int(0.2 * im.width)
        else:
            x = im.width // 2
        # Draw background box
        tw, th = draw.textbbox((0, 0), text, font=font_ly)[2:]
        pad = 10
        bx = (im.width - tw) // 2 - pad
        by = im.height - 100 - pad
        draw.rectangle((bx, by, bx + tw + 2 * pad, by + th + 2 * pad), fill=(0, 0, 0, 128))
        draw.text((x - tw // 2, im.height - 100), text, font=font_ly, fill=(255, 255, 255))
    return im


# ============ Streamlit UI ============
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)

# Sidebar inputs
with st.sidebar:
    st.header("Thiết lập")
    groom = st.text_input("Tên chú rể", "")
    bride = st.text_input("Tên cô dâu", "")
    date_str = st.text_input("Ngày cưới (tuỳ chọn)", "")
    names_line = (groom + " & " + bride).strip()
    if date_str.strip():
        names_line += f"  •  {date_str.strip()}"
    lyric = st.text_input("Lời bài hát / caption", "")
    marquee = st.checkbox("Chạy chữ (marquee)", value=True)

    default_duration = st.number_input("Thời lượng mỗi ảnh (giây)", min_value=2.0, max_value=20.0, value=7.0, step=0.5)

    st.markdown("**Nhạc nền**")
    audio_upload = st.file_uploader("Chọn file nhạc (mp3/m4a/wav)", type=["mp3","m4a","wav"], key="audio")
    use_default_audio = st.checkbox("Dùng nhạc mặc định 'xứng đôi cưới thôi nhen'", value=True)

# Load images from repo folder or upload
st.subheader("Ảnh slideshow")
col1, col2 = st.columns([1, 1])
with col1:
    if st.button("Nạp ảnh từ ./hinhanh"):
        st.session_state["images"] = list_images_from_manifest_or_dir()
    uploaded = st.file_uploader("Hoặc chọn nhiều ảnh", type=["jpg","jpeg","png","webp","gif"], accept_multiple_files=True)
    if uploaded:
        st.session_state["images"] = [{"path": f"uploaded:{i.name}", "effect": "auto", "_data": i.read()} for i in uploaded]

images = st.session_state.get("images", list_images_from_manifest_or_dir())
st.session_state["images"] = images

# Effect table
with col2:
    st.write("### Tuỳ chọn hiệu ứng từng ảnh")
    new_effects = []
    for i, item in enumerate(images):
        name = os.path.basename(item["path"]) if not item["path"].startswith("uploaded:") else item["path"][9:]
        eff_label = st.selectbox(
            f"{i+1}. {name}",
            options=[k for k, _ in EFFECT_OPTIONS],
            format_func=lambda k: dict(EFFECT_OPTIONS)[k],
            index=[k for k, _ in EFFECT_OPTIONS].index(item.get("effect", "auto")),
            key=f"eff_{i}"
        )
        new_effects.append(eff_label)
    for i, eff in enumerate(new_effects):
        images[i]["effect"] = eff

# Preview button
st.divider()

preview_placeholder = st.empty()

def load_image_item(item) -> Image.Image:
    if str(item["path"]).startswith("uploaded:"):
        return Image.open(io.BytesIO(item["_data"]))
    return Image.open(item["path"])  # local file

if st.button("▶ Xem thử (preview)"):
    for i, item in enumerate(images):
        img = load_image_item(item).convert("RGB")
        # determine effect
        eff = item.get("effect", "auto")
        if eff == "auto":
            eff = choose_effect_auto(img)
        # quick preview: 12 frames
        frames = []
        duration = default_duration
        steps = 12
        for f in range(steps):
            t = (f / (steps - 1)) * duration
            frame = render_frame(img, eff, t, duration)
            frame = overlay_texts(frame, names_line, lyric, marquee, t)
            frames.append(frame)
        preview_placeholder.image(frames, caption=f"Ảnh {i+1}/{len(images)} – {eff}")
        time.sleep(0.2)

# Render MP4
st.subheader("Xuất video MP4")
render_clicked = st.button("💾 Render MP4")
progress = st.progress(0)
status = st.empty()

if render_clicked:
    frames_all = []
    total = len(images)
    for i, item in enumerate(images):
        img = load_image_item(item).convert("RGB")
        eff = item.get("effect", "auto")
        if eff == "auto":
            eff = choose_effect_auto(img)
        duration = float(default_duration)
        nframes = int(FPS_EXPORT * duration)
        for f in range(nframes):
            t = (f / max(1, nframes - 1)) * duration
            frame = render_frame(img, eff, t, duration)
            frame = overlay_texts(frame, names_line, lyric, marquee, t)
            frames_all.append(np.array(frame))
        progress.progress(int(((i + 1) / max(1, total)) * 100))
        status.write(f"Đang dựng ảnh {i+1}/{total} – hiệu ứng {eff}")

    clip = ImageSequenceClip(frames_all, fps=FPS_EXPORT)

    audio_path = None
    if audio_upload is not None:
        # save uploaded to temp
        audio_path = "_tmp_audio"
        with open(audio_path, "wb") as f:
            f.write(audio_upload.getvalue())
    elif use_default_audio:
        for cand in DEFAULT_AUDIO_CANDIDATES:
            if os.path.exists(cand):
                audio_path = cand
                break

    if audio_path:
        try:
            ac = AudioFileClip(audio_path)
            # fit audio to clip length (loop or cut)
            if ac.duration < clip.duration:
                # simple loop by concatenating
                loops = int(math.ceil(clip.duration / ac.duration))
                ac = AudioFileClip(audio_path).fx(lambda gf, t: gf(t % ac.duration))
            clip = clip.set_audio(ac.audio_fadein(0.2).audio_fadeout(0.5))
        except Exception as e:
            st.warning(f"Không thể ghép nhạc: {e}")

    clip.write_videofile(OUTPUT_VIDEO, codec="libx264", audio_codec="aac", fps=FPS_EXPORT)
    status.success("Xuất MP4 xong!")
    with open(OUTPUT_VIDEO, "rb") as f:
        st.download_button("Tải slideshow.mp4", data=f, file_name="slideshow.mp4", mime="video/mp4")

# ============ Simple tests (run in app) ============
with st.expander("✅ Self-tests"):
    def test_choose_effect_mapping():
        # Dark image should prefer zoom-out or fade
        img = Image.new("RGB", (400, 300), (10, 10, 10))
        eff = choose_effect_auto(img)
        st.write("Dark image →", eff)
        assert eff in ("fx-zoom-out", "fx-fade")

        # Bright colorful → tilt/fade/pan
        img2 = Image.new("RGB", (400, 300), (250, 120, 120))
        eff2 = choose_effect_auto(img2)
        st.write("Brightish image →", eff2)
        assert eff2 in ("fx-tilt", "fx-fade", "fx-pan-left", "fx-pan-right")

    def test_overlay_newlines():
        img = Image.new("RGB", (800, 450), (0, 0, 0))
        out = overlay_texts(img, "A\n&B\n2026", "", False, 0)
        st.write("Overlay newline handled (no crash)")
        assert isinstance(out, Image.Image)

    try:
        test_choose_effect_mapping()
        test_overlay_newlines()
        st.success("Self-tests passed")
    except AssertionError as e:
        st.error(f"Self-tests failed: {e}")