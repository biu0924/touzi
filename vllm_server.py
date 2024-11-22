from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Optional
from vllm import LLM, SamplingParams
import uvicorn
import time
from datetime import datetime

app = FastAPI(title="Local LLM API Service")

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.7
    top_k: Optional[int] = 10
    top_p: Optional[float] = 0.7
    max_tokens: Optional[int] = 1024

class ChatResponse(BaseModel):
    response: str
    process_time: float

# 添加中间件来记录请求处理时间
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    start_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    response = await call_next(request)
    process_time = time.time() - start_time
    # 打印详细的请求信息和处理时间
    print(f"[{start_datetime}] {request.method} {request.url.path} - Processing Time: {process_time:.2f} seconds")
    return response


def format_chat_prompt(messages: List[ChatMessage]) -> str:
    """
    根据 Qwen 的对话格式构建提示词
    """
    formatted_messages = []

    for msg in messages:
        if msg.role == "system":
            formatted_messages.append(f"<|im_start|>system\n{msg.content}<|im_end|>")
        elif msg.role == "user":
            formatted_messages.append(f"<|im_start|>user\n{msg.content}<|im_end|>")
        elif msg.role == "assistant":
            formatted_messages.append(f"<|im_start|>assistant\n{msg.content}<|im_end|>")
    
    # 添加最后的助手标记来获取回复
    formatted_messages.append("<|im_start|>assistant\n")
    
    return "\n".join(formatted_messages)

# 延迟加载模型
model = None

def get_model():
    global model
    if model is None:
        from vllm import LLM
        model = LLM(
            model="/mnt/workspace/deploy_demo/models/qwen_vl_awq",
            max_model_len=12800,
            gpu_memory_utilization=0.8,  # 控制显存使用率
            tensor_parallel_size=1  # 可以根据多GPU情况调整
        )
    return model
    
@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(request: ChatRequest):
    try:
        start_time = time.time()
        
        # 使用新的提示词格式
        prompt = format_chat_prompt(request.messages)
        
        # 设置采样参数
        sampling_params = SamplingParams(
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            max_tokens=request.max_tokens,
        )

        model = get_model()
        
        generation_start = time.time()
        print(f"[INFO] 模型提示为:\n{prompt}")
        
        outputs = model.generate(prompt, sampling_params)
        response = outputs[0].outputs[0].text.strip()
        
        # 清理响应中可能的标记
        response = response.replace("<|im_end|>", "").strip()
        
        process_time = time.time() - start_time
        generation_time = time.time() - generation_start
        
        print(f"[INFO] 生成完成:")
        print(f"  - 生成时间: {generation_time:.2f} seconds")
        print(f"  - 总处理时间: {process_time:.2f} seconds")
        print(f"  - 响应长度: {len(response)} characters")
        
        return ChatResponse(response=response, process_time=process_time)
        
    except Exception as e:
        error_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[ERROR] [{error_time}] Error during processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("vllm_server:app", host="0.0.0.0", port=8000, reload=True, workers=15)