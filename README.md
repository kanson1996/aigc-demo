# aigc-demo

## 文本生成式AI
1. 后端：搭建server（流式、非流式）
2. 前端： 
   1. 模拟对话框输入
   2. 模拟语音输入
   3. 模拟语音输出
   4. 模拟实时语音交互
3. 模型微调：llama factory

## 图片生成式AI
1. 后端：搭建server（异步）
2. 前端： 
   1. 文件上传、文本输入框
   2. 不同类型模型（checkpoint、lora、controlnet）
   3. 对比SD WebUI、ComfyUI
3. 模型微调：lora


## 语音生成式AI


## 视频生成式AI

## 多模态AI
1. 图文检索

## 判别式AI
```shell
# 初始化安装依赖
pip install -r requirements.txt
# 启动服务端
python server.py
# 启动客户端
python ui.py
```

1. 微调分类模型
   1. 文本（BERT）
   2. 图像（ResNet、ViT）
      1. ViT（Vision Transformer）是一种基于Transformer架构的图像分类模型
   3. 语音

