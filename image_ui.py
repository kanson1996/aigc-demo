import gradio as gr
import requests
import io
from PIL import Image
import base64

def text2img(prompt, negative_prompt, steps, cfg_scale, width, height):
    """文生图函数"""
    # API 端点配置
    url = "http://localhost:8000/sdapi/v1/txt2img"
    
    payload = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "cfg_scale": cfg_scale,
        "width": width,
        "height": height
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        # 解析返回的base64图像
        image_data = base64.b64decode(response.json()['images'][0])
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        return str(e)

def img2img(image, prompt, negative_prompt, denoising_strength, steps):
    """图生图函数"""
    url = "http://localhost:8000/sdapi/v1/img2img"
    
    # 将输入图像转换为base64
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
    
    payload = {
        "init_images": [img_base64],
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "denoising_strength": denoising_strength,
        "steps": steps
    }

    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        
        image_data = base64.b64decode(response.json()['images'][0])
        image = Image.open(io.BytesIO(image_data))
        return image
    except Exception as e:
        return str(e)

# 创建Gradio界面
with gr.Blocks() as demo:
    gr.Markdown("# Stable Diffusion Web UI")
    
    with gr.Tab("文生图"):
        with gr.Row():
            with gr.Column():
                txt2img_prompt = gr.Textbox(label="提示词", lines=3)
                txt2img_negative_prompt = gr.Textbox(label="负面提示词", lines=2)
                txt2img_steps = gr.Slider(minimum=1, maximum=150, value=10, step=1, label="步数")
                txt2img_cfg_scale = gr.Slider(minimum=1, maximum=30, value=7, step=0.5, label="CFG Scale")
                txt2img_width = gr.Slider(minimum=64, maximum=2048, value=512, step=64, label="宽度")
                txt2img_height = gr.Slider(minimum=64, maximum=2048, value=512, step=64, label="高度")
                txt2img_button = gr.Button("生成")
            
            with gr.Column():
                txt2img_output = gr.Image(label="生成结果")
        
        txt2img_button.click(
            fn=text2img,
            inputs=[txt2img_prompt, txt2img_negative_prompt, txt2img_steps, 
                   txt2img_cfg_scale, txt2img_width, txt2img_height],
            outputs=txt2img_output
        )
    
    with gr.Tab("图生图"):
        with gr.Row():
            with gr.Column():
                img2img_input = gr.Image(label="输入图片", type="pil")
                img2img_prompt = gr.Textbox(label="提示词", lines=3)
                img2img_negative_prompt = gr.Textbox(label="负面提示词", lines=2)
                img2img_denoising = gr.Slider(minimum=0, maximum=1, value=0.75, step=0.05, label="重绘幅度")
                img2img_steps = gr.Slider(minimum=1, maximum=150, value=10, step=1, label="步数")
                img2img_button = gr.Button("生成")
            
            with gr.Column():
                img2img_output = gr.Image(label="生成结果")
        
        img2img_button.click(
            fn=img2img,
            inputs=[img2img_input, img2img_prompt, img2img_negative_prompt, 
                   img2img_denoising, img2img_steps],
            outputs=img2img_output
        )

# 启动界面
demo.launch()