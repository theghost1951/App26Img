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
    image_b64 = body.get("image")

    if not prompt:
        return {"error": "prompt required"}

    api_key = os.getenv("STABILITY_API_KEY")

    # Stability requires multipart/form-data ALWAYS
    files = {
        "prompt": (None, prompt),
        "output_format": (None, "png"),
        "model": (None, "sd3.5-large"),
    }

    if image_b64:
        img_bytes = base64.b64decode(image_b64)
        files["image"] = ("ref.png", img_bytes, "image/png")
        files["mode"] = (None, "image-to-image")
        files["strength"] = (None, "0.65")
    else:
        files["mode"] = (None, "text-to-image")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "image/*"
    }

    resp = requests.post(
        "https://api.stability.ai/v2beta/stable-image/generate/sd3",
        headers=headers,
        files=files,
        timeout=60
    )

    if resp.status_code != 200:
        return Response(content=resp.text, media_type="application/json", status_code=resp.status_code)

    return Response(content=resp.content, media_type="image/png")
