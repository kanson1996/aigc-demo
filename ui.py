from io import BytesIO

import gradio as gr
import requests
from PIL import Image
import pandas as pd


def classify_image(image: Image.Image):
    if not image:
        return []
    # 将PIL Image转换为字节流
    img_byte_arr = BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # 创建文件对象
    files = {
        'file': ('image.png', img_byte_arr, 'image/png')
    }
    
    response = requests.post(
        "http://0.0.0.0:8000/classify",
        files=files
    )
    
    if response.status_code == 200:
        results = response.json()['predictions']
        # 转换为DataFrame
        df = pd.DataFrame(results)
        df['score'] = df['score'].apply(lambda x: f"{x*100:.2f}%")
        return df[['label', 'score']].values.tolist()  # 转换为列表格式
    else:
        print("response", response.status_code, response.content.decode())
        return []


with gr.Blocks() as demo:
    gr.Markdown("Image Classify Demo")
    with gr.Column():
        with gr.Row():
            with gr.Column():
                with gr.Column():
                    image = gr.Image(type='pil', height=512)
            with gr.Column():
                smt = gr.Button("开始识别")
                clr = gr.ClearButton()
                classification_result = gr.Dataframe(
                    headers=['标签', '置信度'],
                    label="分类结果",
                    row_count=5,  # 设置显示行数
                    col_count=(2, "fixed"),  # 固定两列
                )
                smt.click(classify_image, [image], classification_result)
                clr.add([image, classification_result])
                clr.click()

if __name__ == '__main__':
    demo.title = "Image Classify Demo"
    demo.launch(share=False, server_name="0.0.0.0", server_port=9908)
