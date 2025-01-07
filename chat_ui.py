import gradio as gr
import requests
import json
from typing import List, Dict

API_URL = "http://localhost:8001/v1/chat/completions"

def format_message(role: str, content: str) -> Dict:
    return {"role": role, "content": content}

def chat_stream(message: str, history: List[List[str]]):
    # 构建消息历史
    messages = []
    for user_msg, bot_msg in history:
        messages.append(format_message("user", user_msg))
        if bot_msg:
            messages.append(format_message("assistant", bot_msg))
    messages.append(format_message("user", message))
    
    # 发送流式请求
    response = requests.post(
        API_URL,
        json={
            "messages": messages,
            "stream": True
        },
        stream=True
    )
    
    partial_message = ""
    
    # 处理流式响应
    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith("data: "):
                if line == "data: [DONE]":
                    break
                    
                data = json.loads(line[6:])
                content = data["choices"][0]["delta"].get("content", "")
                partial_message += content
                yield partial_message

def chat_normal(message: str, history: List[List[str]]):
    # 构建消息历史
    messages = []
    for user_msg, bot_msg in history:
        messages.append(format_message("user", user_msg))
        if bot_msg:
            messages.append(format_message("assistant", bot_msg))
    messages.append(format_message("user", message))
    
    # 发送非流式请求
    response = requests.post(
        API_URL,
        json={
            "messages": messages,
            "stream": False
        }
    )
    
    bot_message = response.json()["choices"][0]["message"]["content"]
    print("bot_message", bot_message)
    # 修改返回值为生成器
    yield bot_message

# 创建Gradio界面
with gr.Blocks(css="#chatbot {height: 500px} .overflow-y-auto {height: 500px}") as demo:
    gr.Markdown("# AI聊天助手")
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.ChatInterface(
                fn=chat_stream,  # 默认使用流式输出
                # title="AI助手",
                description="我是一个AI助手，请问有什么可以帮您？",
                examples=["你好，请介绍一下你自己", "讲个笑话", "写一首诗"],
            )
        
        with gr.Column(scale=1):
            stream_mode = gr.Checkbox(
                label="启用流式输出",
                value=True,
                interactive=True
            )
    
    def switch_mode(stream):
        """切换流式/非流式模式"""
        chatbot.fn = chat_stream if stream else chat_normal
    
    stream_mode.change(
        switch_mode,
        inputs=[stream_mode],
    )

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", share=False, server_port=7860)