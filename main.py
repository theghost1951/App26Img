import os, requests
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import Response, JSONResponse

app = FastAPI()

class Req(BaseModel):
    prompt: str
    model: str = "stabilityai/stable-diffusion-2-1"
    width: int = 768
    height: int = 768

@app.post("/generate")
def generate(r: Req):
    token = os.getenv("HF_TOKEN")
    if not token:
        return JSONResponse({"error":"HF_TOKEN not set in Render > Environment"}, status_code=500)

    # Old inference API - FREE and supports SD 2.1 with no fal.ai
    url = f"https://api-inference.huggingface.co/models/{r.model}"
    headers = {"Authorization": f"Bearer {token}"}
    payload = {"inputs": r.prompt}

    resp = requests.post(url, headers=headers, json=payload, timeout=120)

    # HF returns raw JPEG when success, JSON when error
    ctype = resp.headers.get("content-type","")
    if resp.ok and "image" in ctype:
        return Response(content=resp.content, media_type="image/jpeg")
    
    # Sometimes returns image even with octet-stream
    if resp.ok and len(resp.content) > 10000:
        return Response(content=resp.content, media_type="image/jpeg")

    return JSONResponse({"error": resp.text[:1500]}, status_code=500)

@app.get("/")
def home():
    return {"ok": True, "model": "stabilityai/stable-diffusion-2-1"}
