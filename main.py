import os, tempfile, torch, streamlit as st
os.environ["HY3DGEN_NO_MESH_CLEANUP"] = "1"

st.set_page_config(page_title="Hunyuan3D Generator", layout="centered")
st.title("🌀 Hunyuan3D Web UI")

@st.cache_resource(show_spinner=False)
def load_pipe():
    from hy3dgen.shapegen.pipelines import Hunyuan3DDiTFlowMatchingPipeline
    pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
        "tencent/Hunyuan3D-2mini",
        subfolder="hunyuan3d-dit-v2-mini-turbo",
        use_safetensors=True
    ).to("cpu")
    pipe.to(dtype=torch.float16)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    return pipe

file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if file:
    st.image(file, caption="Preview", use_container_width=True)
    if st.button("🧠 Generate 3D Model"):
        with st.spinner("Generating 3D model... please wait ⏳"):
            pipe = load_pipe()
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tmp.write(file.read())
            tmp.close()
            mesh = pipe(
                image=tmp.name,
                num_inference_steps=8,
                octree_resolution=128,
                output_type="trimesh"
            )[0]
            out_path = "model.stl"
            mesh.export(out_path)

        st.success("✅ 3D model generated!")
        with open(out_path, "rb") as f:
            st.download_button("⬇️ Download model.stl", f, file_name="model.stl")

        try:
            from streamlit_stl import stl_from_file
            st.subheader("🧩 3D Model Preview")
            stl_from_file(out_path, color="#FF9900", auto_rotate=True, height=400)
        except Exception:
            st.info("3D preview unavailable on this environment.")
            