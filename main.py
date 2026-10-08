from fastapi import FastAPI, Request
from fastapi.responses import Response
import base64, io, os
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

app = FastAPI()
pipe_txt = None
pipe_img = None

def get_pipes():
    global pipe_txt, pipe_img
    if pipe_txt is None:
        # tiny-sd = 400MB, fits on Render free tier, 100% free forever
        model_id = "segmind/tiny-sd"
        pipe_txt = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        pipe_img = StableDiffusionImg2ImgPipeline.from_pretrained(model_id, torch_dtype=torch.float32)
        pipe_txt.to("cpu")
        pipe_img.to("cpu")
    return pipe_txt, pipe_img

@app.get("/")
def home():
    return {"status": "FREE Identity Preserving Engine - No Credits"}

@app.post("/generate")
async def generate(request: Request):
    txt_pipe, img_pipe = get_pipes()
    body = await request.json()
    prompt = body.get("prompt","")
    negative = body.get("negative_prompt","")
    strength = float(body.get("strength", 0.35)) # LOW = keep same person
    cfg = float(body.get("cfg_scale", 7.5))
    image_b64 = body.get("image")

    if image_b64:
        # IMAGE-TO-IMAGE - preserves identity when strength is low
        img_bytes = base64.b64decode(image_b64)
        init = Image.open(io.BytesIO(img_bytes)).convert("RGB").resize((512,512))
        # strength 0.25-0.40 = same person, new clothes/pose
        # strength 0.6+ = different person (what you saw)
        result = img_pipe(
            prompt=prompt,
            negative_prompt=negative,
            image=init,
            strength=strength,
            guidance_scale=cfg,
            num_inference_steps=25
        ).images[0]
    else:
        result = txt_pipe(
            prompt=prompt,
            negative_prompt=negative,
            guidance_scale=cfg,
            num_inference_steps=25
        ).images[0]

    buf = io.BytesIO()
    result.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
