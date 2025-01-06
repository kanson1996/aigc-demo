import gradio as gr
import requests
import json
from typing import List, Dict

API_URL = "http://localhost:8001/v1/chat/completions"

def format_message(role: str, content: str) -> Dict:
    return {"role": role, "content": content}

def chat_stream(history: List[List[str]], message: str):
    # 构建消息历史
    messages = []
    for user_msg, bot_msg in history:
        messages.append(format_message("user", user_msg))
        if bot_msg:  # 确保bot消息不为空
            messages.append(format_message("assistant", bot_msg))
    messages.append(format_message("user", message))

    print("messages", messages)
    
    # 发送流式请求
    response = requests.post(
        API_URL,
        json={
            "messages": messages,
            "stream": True
        },
        stream=True
    )
    
    # 初始化bot的回复
    history.append([message, ""])
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
                history[-1][1] = partial_message
                yield history

def chat_normal(history: List[List[str]], message: str):
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
    history.append([message, bot_message])
    return history

# 创建Gradio界面
with gr.Blocks(css="#chatbot {height: 500px} .overflow-y-auto {height: 500px}") as demo:
    gr.Markdown("# AI聊天助手")
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                [],
                elem_id="chatbot",
                height=500,
            )
            with gr.Row():
                msg = gr.Textbox(
                    show_label=False,
                    placeholder="输入您的问题...",
                    container=False
                )
        
        with gr.Column(scale=1):
            stream_mode = gr.Checkbox(
                label="启用流式输出",
                value=True,
                interactive=True
            )
            clear = gr.Button("清空对话")
    
    def user_input(message, history, stream):
        if not message:
            return "", history, gr.update(interactive=True)
        
        fn = chat_stream if stream else chat_normal
        history = fn(history, message) if not stream else history
        
        return "", history, gr.update(interactive=True)
    
    # 移除 reset_textbox 函数，因为我们直接在 user_input 中处理了交互状态
    
    msg.submit(
        user_input,
        inputs=[msg, chatbot, stream_mode],
        outputs=[msg, chatbot, msg]
    )
    
    if stream_mode:  # 如果是流式模式，添加额外的处理
        msg.submit(
            chat_stream,
            inputs=[chatbot, msg],
            outputs=[chatbot],
            queue=False
        )
    
    clear.click(lambda: [], None, chatbot)

if __name__ == "__main__":
    demo.queue()
    demo.launch(server_name="0.0.0.0", share=False, server_port=7860)