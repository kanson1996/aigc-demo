import time
from fastapi import FastAPI, WebSocket
from fastapi.responses import StreamingResponse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Dict
import torch
import json
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from typing import List, Dict, Optional
import json
import asyncio

app = FastAPI()

# 加载模型和分词器
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    messages = body.get("messages", [])
    stream = body.get("stream", False)
    
    # 构建输入上下文
    context = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        context += f"{role}: {content}\n"
    
    # 编码输入
    inputs = tokenizer.encode(context, return_tensors="pt")
    
    if stream:
        return StreamingResponse(
            stream_response(inputs),
            media_type="text/event-stream"
        )
    else:
        # 生成完整回复
        outputs = model.generate(
            inputs,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return JSONResponse({
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "gpt2",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response
                },
                "finish_reason": "stop"
            }]
        })

async def stream_response(inputs):
    try:
        for output in model.generate(
            inputs,
            max_length=200,
            num_return_sequences=1,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True
        ).sequences:
            token = tokenizer.decode(output[-1], skip_special_tokens=True)
            chunk = {
                "id": "chatcmpl-123",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": "gpt2",
                "choices": [{
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "content": token
                    },
                    "finish_reason": None
                }]
            }
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0)
            
        # 发送结束标记
        yield "data: [DONE]\n\n"
            
    except Exception as e:
        print(f"Error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)