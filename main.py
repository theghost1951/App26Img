import os, io, base64, traceback, requests
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import Response, JSONResponse
from PIL import Image

app = FastAPI()

class Req(BaseModel):
    prompt: str
    model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    width: int = 1024
    height: int = 1024

@app.get("/")
def home():
    return {"ok": True, "has_token": bool(os.getenv("HF_TOKEN"))}

@app.post("/generate")
def generate(r: Req):
    try:
        token = os.getenv("HF_TOKEN")
        if not token:
            return JSONResponse({"error": "HF_TOKEN not set"}, status_code=500)

        # Call HF Router directly - no huggingface_hub needed
        url = "https://router.huggingface.co/v1/images/generations"
        headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

        # Try fal-ai first, then together
        last_err = ""
        for provider in ["fal-ai", "together"]:
            try:
                payload = {
                    "model": r.model,
                    "prompt": r.prompt,
                    "provider": provider,
                }
                resp = requests.post(url, json=payload, headers=headers, timeout=120)
                if not resp.ok:
                    last_err = f"{provider} {resp.status_code}: {resp.text[:500]}"
                    continue

                data = resp.json()
                # New API returns b64_json
                if "data" in data and len(data["data"]) > 0:
                    b64 = data["data"][0].get("b64_json")
                    if b64:
                        img_bytes = base64.b64decode(b64)
                        return Response(content=img_bytes, media_type="image/jpeg")
                    img_url = data["data"][0].get("url")
                    if img_url:
                        img_resp = requests.get(img_url, timeout=60)
                        return Response(content=img_resp.content, media_type="image/jpeg")

                last_err = f"{provider} no image in {str(data)[:500]}"
            except Exception as e:
                last_err = f"{provider} exc: {e}"
                continue

        return JSONResponse({"error": f"All failed: {last_err}"}, status_code=500)

    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return JSONResponse({"error": str(e)}, status_code=500)
