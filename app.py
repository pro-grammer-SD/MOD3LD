from fastapi import FastAPI, UploadFile
from fastapi.responses import FileResponse
from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
import torch, tempfile, os

app = FastAPI()

pipe = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained(
    "tencent/Hunyuan3D-2mini",
    subfolder="hunyuan3d-dit-v2-mini-turbo",
    use_safetensors=True
).to("cpu")
pipe.to(dtype=torch.float32)
pipe = torch.compile(pipe, mode="reduce-overhead")
torch.set_num_threads(8)
torch.set_num_interop_threads(8)

@app.post("/generate")
async def generate(file: UploadFile):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        tmp.write(await file.read())
        tmp.flush()
        mesh = pipe(
            image=tmp.name,
            num_inference_steps=10,
            octree_resolution=256,
            output_type="trimesh"
        )[0]
        out_path = "output.glb"
        mesh.export(out_path)
    return FileResponse(out_path, filename="model.glb")