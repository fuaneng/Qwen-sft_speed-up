# app.py
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field
from typing import List, Dict, Union
import uvicorn
import logging

# 从 inference_engine.py 导入模型加载和推理函数
from inference_engine import load_model, generate_response, model, processor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 初始化 FastAPI 应用
app = FastAPI(
    title="Qwen2.5-VL 图像理解 API",
    description="基于 Qwen2.5-VL 的多模态问答接口，支持图片 URL 和文本输入。"
)

# --- 应用事件：启动时加载模型 ---
@app.on_event("startup")
async def startup_event():
    """
    FastAPI 应用程序启动时执行的事件。
    """
    logging.info("FastAPI 应用程序启动中，正在加载模型...")
    try:
        load_model()
        logging.info("模型加载成功。")
    except Exception as e:
        logging.critical(f"模型加载失败，应用程序将无法正常工作: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"模型加载失败: {e}"
        )

# --- Pydantic 模型定义 ---
class MessageContent(BaseModel):
    """
    消息内容的结构，可以包含文本或图片 URL。
    """
    type: str = Field(..., description="内容类型，'text' 或 'image'")
    text: Union[str, None] = Field(None, description="文本内容，当 type 为 'text' 时使用")
    image: Union[str, None] = Field(None, description="图片内容，URL 字符串，当 type 为 'image' 时使用")

class Message(BaseModel):
    """
    聊天消息的结构。
    """
    role: str = Field(..., description="消息角色，例如 'user'")
    content: List[MessageContent] = Field(..., description="消息内容列表，遵循 Qwen-VL 格式")

class InferenceRequest(BaseModel):
    """
    推理请求的输入模型。
    """
    messages: List[Message] = Field(..., description="遵循 Qwen 聊天模板格式的消息列表")

class InferenceResponse(BaseModel):
    """
    推理请求的输出模型。
    """
    result: str = Field(..., description="模型生成的文本结果")

# --- API 路由定义 ---
@app.post("/v1/chat/completions", response_model=InferenceResponse,
          summary="执行多模态聊天补全",
          description="接收包含图片 URL 和文本的请求，返回模型生成的响应。")
async def chat_completions(request: InferenceRequest):
    """
    处理多模态聊天补全请求。
    """
    logging.info("收到新的 /v1/chat/completions 请求。")
    try:
        raw_messages = []
        for msg in request.messages:
            content_list = [block.model_dump(exclude_none=True) for block in msg.content]
            raw_messages.append({"role": msg.role, "content": content_list})

        logging.debug(f"即将发送给推理引擎的消息: {raw_messages}")
        result = generate_response(raw_messages)
        logging.info("请求处理完成。")
        return InferenceResponse(result=result)
    except Exception as e:
        logging.error(f"处理请求时发生错误: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"内部服务器错误: {e}"
        )

@app.get("/health", summary="健康检查", response_model=Dict[str, Union[str, bool]])
def health_check():
    """
    检查 API 服务的健康状态和模型加载情况。
    """
    model_loaded = model is not None and processor is not None
    if model_loaded:
        return {"status": "ok", "message": "服务正常运行，模型已加载。", "model_loaded": True}
    else:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail={"status": "unavailable", "message": "服务正在启动或模型加载失败。", "model_loaded": False}
        )