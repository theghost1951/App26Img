from fastapi import FastAPI, Request
from fastapi.responses import Response
import requests, os, base64

app = FastAPI()

@app.get("/")
def home(): return {"status": "Agnes 2.5 Flash Ready"}

@app.post("/generate")
async def generate(request: Request):
    body = await request.json()
    prompt = body.get("prompt")
    negative = body.get("negative_prompt","")
    aspect = body.get("aspect_ratio","1:1")
    cfg = body.get("cfg_scale", 9)
    strength = body.get("strength", 0.35)
    image_b64 = body.get("image")

    api_key = os.getenv("STABILITY_API_KEY")
    
    files = {
        "prompt": (None, prompt),
        "negative_prompt": (None, negative),
        "output_format": (None, "png"),
        "model": (None, "sd3.5-large"),
        "cfg_scale": (None, str(cfg)),
    }

    if image_b64:
        # IMAGE-TO-IMAGE - NO aspect_ratio allowed!
        img_bytes = base64.b64decode(image_b64)
        files["image"] = ("ref.png", img_bytes, "image/png")
        files["mode"] = (None, "image-to-image")
        files["strength"] = (None, str(strength))
    else:
        # TEXT-TO-IMAGE - aspect_ratio allowed
        files["mode"] = (None, "text-to-image")
        files["aspect_ratio"] = (None, aspect)

    headers = {"Authorization": f"Bearer {api_key}", "Accept": "image/*"}
    resp = requests.post("https://api.stability.ai/v2beta/stable-image/generate/sd3", headers=headers, files=files, timeout=90)
    
    if resp.status_code != 200:
        return Response(content=resp.text, media_type="application/json", status_code=resp.status_code)
    return Response(content=resp.content, media_type="image/png")
