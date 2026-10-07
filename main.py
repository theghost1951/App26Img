from fastapi import FastAPI, Request
from fastapi.responses import Response
import requests
import os
import base64

app = FastAPI()

@app.get("/")
def home():
    return {"status": "Stability API Ready"}

@app.post("/generate")
async def generate(request: Request):
    body = await request.json()
    prompt = body.get("prompt")
    image_b64 = body.get("image")  # optional reference image

    if not prompt:
        return {"error": "prompt required"}

    api_key = os.getenv("STABILITY_API_KEY")
    
    url = "https://api.stability.ai/v2beta/stable-image/generate/sd3"
    
    files = {}
    data = {
        "prompt": prompt,
        "output_format": "png",
        "model": "sd3.5-large",
        "mode": "text-to-image"
    }

    if image_b64:
        # img2img mode
        img_bytes = base64.b64decode(image_b64)
        files["image"] = ("ref.png", img_bytes, "image/png")
        data["mode"] = "image-to-image"
        data["strength"] = "0.65"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "image/*"
    }

    resp = requests.post(url, headers=headers, files=files, data=data, timeout=60)
    
    if resp.status_code != 200:
        return {"error": resp.text}

    return Response(content=resp.content, media_type="image/png")
