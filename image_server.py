import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from PIL import Image
import io
import uvicorn
import base64

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')
model_id = "CompVis/stable-diffusion-v1-4"

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
t2i_pipe = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.bfloat16).to(device)
i2i_pipe = StableDiffusionImg2ImgPipeline(**t2i_pipe.components).to(device)

app = FastAPI()


@app.post("/sdapi/v1/txt2img")
async def txt2img(request: Request):
    data = await request.json()
    image = t2i_pipe(
        data['prompt'],
        negative_prompt=data['negative_prompt'],
        height=data['height'],
        width=data['width'],
        guidance_scale=data['cfg_scale'],
        num_inference_steps=data['steps'],
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    
    # 将PIL Image转换为base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return JSONResponse({"images": [img_str]})


@app.post("/sdapi/v1/img2img")
async def img2img(request: Request):
    data = await request.json()
    img_data = base64.b64decode(data['init_images'][0])
    
    # 将解码后的数据转换为PIL Image
    init_image = Image.open(io.BytesIO(img_data))

    image = i2i_pipe(
        prompt=data['prompt'],
        image=init_image,
        negative_prompt=data['negative_prompt'],
        strength=data['denoising_strength'],
        num_inference_steps=data['steps'],
        max_sequence_length=512,
        generator=torch.Generator("cpu").manual_seed(0)
    ).images[0]
    
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return JSONResponse({"images": [img_str]})



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

