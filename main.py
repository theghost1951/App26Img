import os, io
from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import InferenceClient
from fastapi.responses import Response

app = FastAPI()
# Set HF_TOKEN in Render dashboard
client = InferenceClient(token=os.getenv("HF_TOKEN"))

class Req(BaseModel):
    prompt: str
    model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    width: int = 1024
    height: int = 1024

@app.get("/")
def home():
    return {"ok": True}

@app.post("/generate")
def generate(r: Req):
    # fal-ai provider works for both SDXL and FLUX
    provider = "fal-ai"
    if "flux" in r.model.lower():
        provider = "fal-ai"  # or "together"
    
    img = client.text_to_image(
        r.prompt,
        model=r.model,
        provider=provider,
        width=r.width,
        height=r.height
    )
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return Response(content=buf.getvalue(), media_type="image/jpeg")
