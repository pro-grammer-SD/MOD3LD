import streamlit as st
import requests, tempfile

st.set_page_config(page_title="Hunyuan3D Mini Turbo", layout="centered")
st.title("🧱 Hunyuan3D Mini-Turbo (CPU Host)")

uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
if uploaded:
    st.info("Generating 3D model... please wait ⏳")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(uploaded.read())
        tmp.flush()
        res = requests.post("http://localhost:8080/generate", files={"file": open(tmp.name, "rb")})
    if res.status_code == 200:
        with open("model.glb", "wb") as f:
            f.write(res.content)
        st.success("✅ 3D model ready!")
        st.download_button("Download model.glb", res.content, "model.glb")
    else:
        st.error("Generation failed. Try again later.")