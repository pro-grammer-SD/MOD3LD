import streamlit as st
import requests
import tempfile
import threading

st.set_page_config(page_title="Hunyuan3D Generator", page_icon="🧊", layout="centered")
st.title("🧊 Hunyuan3D Web UI")

if "file" not in st.session_state:
    st.session_state.file = None
if "generating" not in st.session_state:
    st.session_state.generating = False
if "stop" not in st.session_state:
    st.session_state.stop = False

def cancel_generation():
    st.session_state.stop = True
    st.session_state.generating = False

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_file:
    st.session_state.file = uploaded_file
    st.image(uploaded_file, caption="Preview", use_container_width=True)
    if st.button("❌ Remove Image"):
        st.session_state.file = None
        st.experimental_rerun()

if st.session_state.file:
    col1, col2 = st.columns(2)
    with col1:
        generate = st.button("🚀 Generate 3D Model")
    with col2:
        cancel = st.button("🛑 Cancel Generation", on_click=cancel_generation)

    if generate and not st.session_state.generating:
        st.session_state.generating = True
        st.session_state.stop = False
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(st.session_state.file.read())
            tmp.flush()
            with st.spinner("Generating 3D model... please wait (CPU mode, might take a bit)"):
                try:
                    res = requests.post("https://mod3ld.onrender.com/generate", files={"file": open(tmp.name, "rb")}, timeout=600)
                    if st.session_state.stop:
                        st.warning("Generation canceled.")
                    elif res.status_code == 200:
                        with open("model.glb", "wb") as f:
                            f.write(res.content)
                        st.success("✅ Model generated successfully!")
                        with open("model.glb", "rb") as f:
                            st.download_button("💾 Download .glb", f, file_name="model.glb")
                    else:
                        st.error(f"Generation failed: {res.status_code}")
                except Exception as e:
                    if not st.session_state.stop:
                        st.error(f"Error: {e}")
                finally:
                    st.session_state.generating = False
                    