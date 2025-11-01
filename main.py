import streamlit as st
import torch, tempfile, os
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
import trimesh
from streamlit_stl import stl_from_file

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
                out_path = "model.stl"
                mesh.export(out_path)

            st.success("✅ 3D model generated!")
            st.download_button("⬇️ Download model.stl", open(out_path, "rb"), file_name="model.stl")

            st.subheader("🧩 3D Model Preview")
            stl_from_file(
                file_path=out_path,
                color="#FF9900",
                material="material",
                auto_rotate=True,
                opacity=1,
                shininess=100,
                cam_v_angle=60,
                cam_h_angle=-90,
                height=500,
                max_view_distance=1000,
            )
            