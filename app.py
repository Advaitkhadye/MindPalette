import os, requests, streamlit as st, base64, zipfile
from io import BytesIO
from PIL import Image
from datetime import datetime
from dotenv import load_dotenv
from transformers import pipeline

# ── Load API Key ──
load_dotenv()
STABILITY_API_KEY = os.getenv("STABILITY_API_KEY")

# ── Prompt Enhancer ───
@st.cache_resource
def load_enhancer():
    return pipeline("text-generation", model="gpt2")
enhancer = load_enhancer()

# ── Streamlit Setup ──
st.set_page_config(page_title="MindPalette", layout="wide")
st.title(" MindPalette  - AI Image Generator")

#  Load External CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")



# ── Session State ──
if "gallery" not in st.session_state: st.session_state.gallery = []
if "last_image" not in st.session_state: st.session_state.last_image = None
if "enhanced_prompt" not in st.session_state: st.session_state.enhanced_prompt = None

# ── Prompt Input ──
default = "Batman standing on Gotham City rooftop, cinematic"
prompt = st.text_area("Enter your prompt:", st.session_state.get("enhanced_prompt", default))

# ── Art Style  ────
style = st.selectbox("Choose Art Style 🎭",
    ["None","Anime","Cyberpunk","Realistic","Oil Painting","Sketch","Pixar-style"])
if style != "None":
    prompt = f"{prompt}, in {style} style"

# ── Prompt Enhancer (Sidebar) ───
with st.sidebar:
    st.markdown("### 🤖 Prompt Enhancer")
    idea = st.text_input("Short idea:", "boy studying at desk")
    if st.button("Enhance Prompt"):
        with st.spinner("Enhancing..."):
            raw = enhancer(
                f"Improve this art prompt in one short sentence: {idea}",
                max_length=25, num_return_sequences=1
            )[0]["generated_text"]
            # keep ONLY the improved sentence, remove the leading instruction if model echoes it
            short = raw.replace("Improve this art prompt in one short sentence:", "").strip()
            short = short.split(".")[0].strip()
            st.session_state.enhanced_prompt = short

# ── Floating suggestion bubble ────
    st.markdown(
        f'<div class="bubble">👉 Suggested: <b>{st.session_state.enhanced_prompt}</b></div>',
        unsafe_allow_html=True
    )

# ── Image Generation via Stability ───
def generate_image(prompt, w=1024, h=1024):
    if not STABILITY_API_KEY:
        st.error("⚠️ Missing Stability API Key in .env")
        return None
    r = requests.post(
        "https://api.stability.ai/v1/generation/stable-diffusion-xl-1024-v1-0/text-to-image",
        headers={
            "Content-Type":"application/json",
            "Accept":"application/json",
            "Authorization": f"Bearer {STABILITY_API_KEY}"
        },
        json={
            "text_prompts":[{"text":prompt}],
            "cfg_scale":12,"samples":1,"steps":40,
            "width":w,"height":h
        }
    )
    if r.status_code == 200:
        img64 = r.json()["artifacts"][0]["base64"]
        return Image.open(BytesIO(base64.b64decode(img64)))
    else:
        st.error(f"❌ Error {r.status_code}: {r.text}")
        return None

# ── Actions-Buttons  ──
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("🎨 Generate"):
        with st.spinner("Generating..."):
            img = generate_image(prompt)
            if img:
                st.session_state.last_image = img
                st.session_state.gallery.append({
                    "prompt": prompt,
                    "image": img,
                    "time": datetime.now().strftime("%H:%M:%S")
                })
                st.success("✅ Done!")

with col2:
    if st.button("✨ Variations"):
        if st.session_state.last_image:
            with st.spinner("Making variations..."):
                for _ in range(2):
                    v = generate_image(prompt)
                    if v:
                        st.session_state.gallery.append({
                            "prompt": prompt + " (var)",
                            "image": v,
                            "time": datetime.now().strftime("%H:%M:%S")
                        })
            st.success("✅ Variations Ready")

with col3:
    if st.button("⬆️ Upscale 2K"):
        if st.session_state.last_image:
            orig = st.session_state.last_image
            up = orig.resize((2048, 2048))
            st.subheader("🔍 Before / After")
            colA, colB = st.columns(2)
            with colA: st.image(orig, caption="Before", use_container_width=True)
            with colB: st.image(up, caption="After", use_container_width=True)
            st.session_state.gallery.append({
                "prompt": prompt + " (upscaled)",
                "image": up,
                "time": datetime.now().strftime("%H:%M:%S")
            })
            st.success("✅ Upscaled")

# ── Gallery with per image download and ZIP ──
if st.session_state.gallery:
    st.subheader("🖼️ Gallery")
    cols = st.columns(3)
    for i, item in enumerate(st.session_state.gallery):
        with cols[i % 3]:
            st.image(item["image"],
                     caption=f"{item['prompt']} ({item['time']})",
                     use_container_width=True)
            buf = BytesIO(); item["image"].save(buf, "PNG")
            st.download_button("⬇️ Download",
                               buf.getvalue(),
                               file_name=f"mindpalette_{i+1}.png",
                               mime="image/png")

    zip_buf = BytesIO()
    with zipfile.ZipFile(zip_buf, "w") as zf:
        for i, item in enumerate(st.session_state.gallery):
            img_buf = BytesIO(); item["image"].save(img_buf, "PNG")
            zf.writestr(f"mindpalette_{i+1}.png", img_buf.getvalue())
    st.download_button("⬇️ Download All as ZIP",
                       zip_buf.getvalue(),
                       "mindpalette_gallery.zip",
                       "application/zip")
