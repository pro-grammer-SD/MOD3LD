import os, psutil, shutil, tempfile, torch, streamlit as st, time, pandas as pd
import plotly.graph_objects as go

os.environ["HY3DGEN_NO_MESH_CLEANUP"] = "1"

st.set_page_config(page_title="Hunyuan3D Generator", layout="centered")
st.title("🌀 Hunyuan3D Web UI")

def system_specs():
    cpu = os.cpu_count()
    ram = psutil.virtual_memory()
    disk = shutil.disk_usage("/")
    torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "CPU Cores": cpu,
        "RAM Used": round(ram.used / (1024**3), 2),
        "RAM Total": round(ram.total / (1024**3), 2),
        "Storage Free": round(disk.free / (1024**3), 2),
        "Storage Total": round(disk.total / (1024**3), 2),
        "Torch Device": str(torch_device).upper(),
    }

placeholder = st.empty()
chart_placeholder = st.empty()
data = pd.DataFrame(columns=["Time", "CPU (%)", "RAM (GB Used)", "Storage (GB Free)"])

for _ in range(20):
    specs = system_specs()
    now = time.strftime("%H:%M:%S")
    data.loc[len(data)] = [
        now,
        psutil.cpu_percent(),
        specs["RAM Used"],
        specs["Storage Free"],
    ]
    placeholder.markdown(
        f"""### 🧠 System Info
**CPU Cores:** {specs["CPU Cores"]}
**RAM:** {specs["RAM Used"]} GB / {specs["RAM Total"]} GB  
**Storage:** {specs["Storage Free"]} GB / {specs["Storage Total"]} GB  
**Torch Device:** {specs["Torch Device"]}
"""
    )
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data["Time"], y=data["CPU (%)"], mode="lines+markers", name="CPU (%)"))
    fig.add_trace(go.Scatter(x=data["Time"], y=data["RAM (GB Used)"], mode="lines+markers", name="RAM Used (GB)"))
    fig.add_trace(go.Scatter(x=data["Time"], y=data["Storage (GB Free)"], mode="lines+markers", name="Storage Free (GB)"))
    fig.update_layout(margin=dict(l=10, r=10, t=20, b=10), height=250, legend=dict(orientation="h"))
    chart_placeholder.plotly_chart(fig, use_container_width=True)
    time.sleep(1)

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
    st.image(file, caption="Preview", width='stretch')
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
            