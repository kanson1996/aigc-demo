import time
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from typing import List, Dict
import json
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, JSONResponse
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Optional
import json
import asyncio

app = FastAPI()

# 加载模型和分词器
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2-1.5B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B-Instruct")


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
    
    # 编码输入并设置attention mask
    inputs = tokenizer(context, return_tensors="pt", padding=True)
    
    if stream:
        return StreamingResponse(
            stream_response(inputs),
            media_type="text/event-stream"
        )
    else:
        # 生成完整回复，添加更多控制参数
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=2048,  # 增加最大长度
            min_length=50,    # 设置最小长度
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,        # 添加top_p采样
            repetition_penalty=1.2,  # 添加重复惩罚
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True    # 启用采样
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
        # 一次性生成完整序列
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=2048,
            min_length=50,
            num_return_sequences=1,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            return_dict_in_generate=True,
            output_scores=True
        )
        
        # 获取生成的序列
        generated_sequence = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        
        # 逐个token输出
        for token_id in generated_sequence:
            token = tokenizer.decode([token_id], skip_special_tokens=True)
            if token.strip():  # 只发送非空token
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
                await asyncio.sleep(0.01)  # 添加小延迟使流更平滑
            
        # 发送结束标记
        yield "data: [DONE]\n\n"
            
    except Exception as e:
        print(f"Error: {e}")
        yield f"data: {json.dumps({'error': str(e)})}\n\n"


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)