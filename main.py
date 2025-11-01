import os
import shutil
import psutil
import torch
import streamlit as st

st.set_page_config(page_title="Hunyuan3D Generator", layout="centered")
st.title("🌀 Hunyuan3D Web UI")

total_ram = round(psutil.virtual_memory().total / (1024 ** 3), 2)
used_ram = round(psutil.virtual_memory().used / (1024 ** 3), 2)
disk_total = round(shutil.disk_usage("/").total / (1024 ** 3), 2)
disk_free = round(shutil.disk_usage("/").free / (1024 ** 3), 2)
cpu_cores = os.cpu_count()

st.sidebar.header("🧠 System Info")
st.sidebar.markdown(f"**CPU Cores:** {cpu_cores}")
st.sidebar.markdown(f"**RAM:** {used_ram} GB / {total_ram} GB")
st.sidebar.markdown(f"**Storage Free:** {disk_free} GB / {disk_total} GB")
st.sidebar.markdown(f"**Torch Device:** {'CUDA' if torch.cuda.is_available() else 'CPU'}")
