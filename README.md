# aigc-demo

## 文本生成式AI
```shell
# 初始化安装依赖
pip install -r requirements.txt
# 启动服务端
python chat_server.py
# 启动客户端
python chat_ui.py
```
1. 后端：搭建server（流式、非流式）
   1. huggingface模型：gpt2
   2. 推理类型：generate（续写生成）、chat（对话生成）
2. 前端： 
   1. 模拟对话框输入
   2. 模拟语音输入
   3. 模拟语音输出
   4. 模拟实时语音交互
3. 模型微调：llama factory
4. 推理加速：
5. 模型量化：
6. 硬件加速：
7. 开源框架：Ollana

## 图片生成式AI
```shell
# 初始化安装依赖
pip install -r requirements.txt
# 启动服务端
python image_server.py
# 启动客户端
python image_ui.py
```
1. 后端：搭建server（异步）
2. 前端： 
   1. 文件上传、文本输入框
   2. 不同类型模型（checkpoint、lora、controlnet）
   3. 开源框架：SD WebUI、ComfyUI
3. huggingface模型：CompVis--stable-diffusion-v1-4
4. 模型微调：lora
5. 模型类型：文生图、图生图、图像重绘
6. 应用：工作流


## 语音生成式AI
1. AI音乐：sino

## 视频生成式AI
1. Sora
2. HunyuanVideo
3. CogVideoX

## 多模态AI
1. 图文检索

## AI Agent
1. 开源框架：langchain、dify
2. AutoGPT、FastGPT
3. Cursor、Devin

## 端侧小模型
1. MiniCPM


## AI Native
1. 协议：MCP（Model Context Protocol）


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

