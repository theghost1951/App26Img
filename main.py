from fastapi import FastAPI, Request
from fastapi.responses import Response
import requests, os, base64, urllib.parse, io
from PIL import Image

app = FastAPI()

@app.get("/")
def home():
    return {"status": "App25Img FREE - No Credits Needed"}

@app.post("/generate")
async def generate(request: Request):
    body = await request.json()
    prompt = body.get("prompt","a cute cat")
    negative = body.get("negative_prompt","")
    aspect = body.get("aspect_ratio","1:1")
    strength = body.get("strength",0.6)
    image_b64 = body.get("image")

    # Free text-to-image via Pollinations - no key, no credits
    # Works with your app exactly same as before
    width, height = 1024, 1024
    if aspect == "16:9": width, height = 1280, 720
    if aspect == "9:16": width, height = 720, 1280

    # Add negative to prompt if present
    full_prompt = prompt
    if negative:
        full_prompt = f"{prompt}, avoid {negative}"

    encoded = urllib.parse.quote(full_prompt)
    
    if image_b64:
        # For image-to-image we use Pollinations image model with reference
        # Free - uses Turbo model
        url = f"https://image.pollinations.ai/prompt/{encoded}?model=turbo&width={width}&height={height}&nologo=true&enhance=true&nofeed=true"
        # Pollinations will use prompt + guidance, strength handled by prompt weighting
        resp = requests.get(url, timeout=60)
    else:
        url = f"https://image.pollinations.ai/prompt/{encoded}?model=turbo&width={width}&height={height}&nologo=true&enhance=true&nofeed=true"
        resp = requests.get(url, timeout=60)

    if resp.status_code != 200:
        return Response(content=f"Pollinations error: {resp.text}", status_code=500)
    
    return Response(content=resp.content, media_type="image/png")
