from transformers import pipeline
from fastapi import FastAPI, UploadFile, File
from PIL import Image
import io
import uvicorn

# 创建FastAPI应用
app = FastAPI()

# 初始化ViT图像分类pipeline
classifier = pipeline("image-classification", model="google/vit-base-patch16-224")

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    # 读取上传的图片
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))
    
    # 使用ViT模型进行预测
    results = classifier(image)
    
    # 返回预测结果
    return {
        "predictions": results
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)