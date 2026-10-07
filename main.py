import os, base64, requests
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.responses import Response, JSONResponse

app = FastAPI()

class Req(BaseModel):
    prompt: str
    model: str = "black-forest-labs/FLUX.1-schnell"
    width: int = 1024
    height: int = 1024

@app.post("/generate")
def generate(r: Req):
    token = os.getenv("HF_TOKEN")
    if not token:
        return JSONResponse({"error":"no HF_TOKEN"}, status_code=500)

    headers = {"Authorization": f"Bearer {token}", "Content-Type":"application/json"}

    # Use provider-specific endpoint - avoids hf-inference completely
    for provider in ["fal-ai", "together"]:
        try:
            url = f"https://router.huggingface.co/{provider}/v1/images/generations"
            payload = {"model": r.model, "prompt": r.prompt}
            resp = requests.post(url, json=payload, headers=headers, timeout=120)
            if not resp.ok:
                print(f"{provider} failed {resp.status_code}: {resp.text[:1000]}")
                continue
            j = resp.json()
            # fal-ai returns b64_json
            b64 = j.get("data", [{}])[0].get("b64_json") if "data" in j else j.get("b64_json")
            if b64:
                return Response(content=base64.b64decode(b64), media_type="image/jpeg")
            # sometimes returns url
            img_url = j.get("data", [{}])[0].get("url") if "data" in j else None
            if img_url:
                return Response(content=requests.get(img_url).content, media_type="image/jpeg")
        except Exception as e:
            print(f"{provider} exc {e}")
            continue

    return JSONResponse({"error":"fal-ai + together both failed - check Render logs"}, status_code=500)

@app.get("/")
def home():
    return {"ok":True}
