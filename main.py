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
        return JSONResponse({"error":"no HF_TOKEN"}, status_code=500)

    # NEW endpoint - router.huggingface.co resolves on Render
    url = f"https://router.huggingface.co/hf-inference/models/{r.model}"
    headers = {"Authorization": f"Bearer {token}", "Content-Type":"application/json"}
    payload = {"inputs": r.prompt}

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=180)
    except Exception as e:
        return JSONResponse({"error": f"connect fail {e}"}, status_code=500)

    # Success = raw image bytes
    if resp.ok and len(resp.content) > 10000:
        return Response(content=resp.content, media_type="image/jpeg")

    return JSONResponse({"error": resp.text[:2000], "status": resp.status_code}, status_code=500)

@app.get("/")
def home():
    return {"ok": True, "model": "stabilityai/stable-diffusion-2-1"}
