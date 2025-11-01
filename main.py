import streamlit as st
import torch, tempfile, os
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
import trimesh

st.set_page_config(page_title="Hunyuan3D Generator", layout="centered")
st.title("🌀 Hunyuan3D Web UI")

@st.cache_resource
def load_pipe():
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mini",
        subfolder="hunyuan3d-dit-v2-mini-turbo",
        use_safetensors=True
    ).to("cpu")
    pipe.to(dtype=torch.float32)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(8)
    return pipe

pipe = load_pipe()

file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if file:
    st.image(file, caption="Preview", use_container_width=True)
    if st.button("🧠 Generate 3D Model"):
        with st.spinner("Generating 3D model... please wait ⏳"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                tmp.write(file.read())
                mesh = pipe(
                    image=tmp.name,
                    num_inference_steps=10,
                    octree_resolution=256,
                    output_type="trimesh"
                )[0]
                out_path = "model.glb"
                mesh.export(out_path)
            with open(out_path, "rb") as f:
                st.download_button("⬇️ Download model.glb", f, file_name="model.glb")
                