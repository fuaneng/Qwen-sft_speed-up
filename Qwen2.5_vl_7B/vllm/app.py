import base64
import io
import os
import gc
import torch
import requests
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import socket  # 新增: 导入 socket 库

# 确保 qwen_vl_utils.py 在你的项目路径下
# 如果不在，你可能需要使用 sys.path.append() 来添加路径
try:
    from qwen_vl_utils import process_vision_info
except ImportError:
    raise ImportError("错误: 无法导入 'qwen_vl_utils.py'。请确保该文件与 app.py 在同一目录下。")


# --- 1. 全局配置与模型加载 ---
# 在服务启动时只执行一次，避免重复加载

# 动态设置Ray的临时目录，以解决PermissionError问题
# 这段代码必须在vLLM和Ray初始化之前执行
ray_temp_dir = "/DATA/z60046962/Qwen2_5_SFT-main/ray_tmp"
try:
    os.makedirs(ray_temp_dir, exist_ok=True)
    os.environ["RAY_TMPDIR"] = ray_temp_dir
    print(f"Ray临时目录已设置为：{ray_temp_dir}")
except OSError as e:
    print(f"无法创建或设置Ray临时目录：{e}")
    print("请确保你对该目录有写入权限，否则vLLM可能无法正常启动。")

# 合并好的模型路径
MODEL_PATH = r"/DATA/z60046962/Qwen2_5_SFT-main/train/output/Qwen2.5-VL-7B-Instruct-merged-10500"
MAX_TOKENS = 2048  # 模型生成文本的最大长度

print("正在启动服务并加载模型 (vLLM)...")

# 检查模型路径是否存在
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"错误: 模型路径不存在: '{MODEL_PATH}'")

# 配置并加载 vLLM
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=1,  # 如果你有多张GPU，可以设置为多卡并行
    trust_remote_code=True,
    dtype="bfloat16",  # 7B 模型推荐使用 bfloat16
    enforce_eager=True,  # 多模态模型必须使用 eager 模式
    max_model_len=8192,  # 根据模型能力设定最大长度
    gpu_memory_utilization=0.85,  # 根据你的显存调整，如果OOM可以适当调低
)

# 配置采样参数
sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0,
    max_tokens=MAX_TOKENS,
)

# 加载处理器
processor = AutoProcessor.from_pretrained(MODEL_PATH)

print("模型和处理器加载完成，服务已准备就绪。")


# --- 2. FastAPI 应用设置 ---

app = FastAPI(
    title="Qwen-VL 推理服务",
    description="一个基于 Qwen-VL 模型的图文理解 API 服务"
)

# 定义请求体的数据结构
class InferenceRequest(BaseModel):
    prompt: str
    image_url: Optional[str] = None
    image_base64: Optional[str] = None


# --- 3. 辅助函数 ---
# 这些函数是从你的原始脚本迁移过来的

def resize_image(image_bytes, max_dim=512):
    """从字节流缩放图片，返回缩放后的字节流"""
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        width, height = img.size
        if max(width, height) > max_dim:
            if width > height:
                new_width = max_dim
                new_height = int(height * (max_dim / width))
            else:
                new_height = max_dim
                new_width = int(width * (max_dim / height))
            img = img.resize((new_width, new_height), Image.LANCZOS)

        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return buffered.getvalue()
    except Exception as e:
        print(f"图片缩放失败: {e}")
        return None

def prepare_vllm_input(prompt, image_base64):
    """为vLLM准备单条图文输入"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image/png;base64,{image_base64}"},
                {"type": "text", "text": f"评价这张图片:{prompt}"}
            ],
        }
    ]
    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, _ = process_vision_info(messages)
    llm_inputs = {
        "prompt": text_prompt,
        "multi_modal_data": {"image": image_inputs},
    }
    return llm_inputs


# --- 4. API 接口定义 ---

@app.post("/predict")
async def predict(request: InferenceRequest):
    """
    接收图片和文本，返回模型的生成结果。
    - **prompt**: 必需，用户提出的问题。
    - **image_url**: 可选，图片的公开访问URL。
    - **image_base64**: 可选，图片的Base64编码字符串。
    **注意**: `image_url` 和 `image_base64` 必须提供其中一个。
    """
    if not request.image_url and not request.image_base64:
        raise HTTPException(status_code=400, detail="必须提供 'image_url' 或 'image_base64'。")

    image_bytes = None
    image_base64 = None

    # Step 1: 获取图片数据
    if request.image_url:
        try:
            response = requests.get(request.image_url, timeout=20)
            response.raise_for_status()
            image_bytes = response.content
        except requests.exceptions.RequestException as e:
            raise HTTPException(status_code=400, detail=f"从 URL 下载图片失败: {e}")
    else: # request.image_base64
        try:
            image_bytes = base64.b64decode(request.image_base64)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Base64 解码失败: {e}")

    # Step 2: 缩放图片并进行 Base64 编码
    resized_bytes = resize_image(image_bytes)
    if not resized_bytes:
        raise HTTPException(status_code=500, detail="服务器处理图片失败。")
    image_base64 = base64.b64encode(resized_bytes).decode('utf-8')

    # Step 3: 准备模型输入并进行推理
    try:
        llm_input = prepare_vllm_input(request.prompt, image_base64)
        outputs = llm.generate([llm_input], sampling_params=sampling_params)
        result_text = outputs[0].outputs[0].text.strip()
    except Exception as e:
        # 记录详细错误日志
        print(f"模型推理时发生严重错误: {e}")
        # 清理可能的CUDA缓存
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(status_code=500, detail=f"模型推理失败: {e}")

    # Step 4: 返回结果
    return {"result": result_text}

# 添加一个根路径，用于健康检查
@app.get("/")
def read_root():
    return {"status": "Qwen-VL inference service is running."}

# 如果你想直接运行 `python app.py` (不推荐用于生产)
if __name__ == "__main__":
    import uvicorn
    # 获取本机 IP 地址
    try:
        # 连接到一个外部地址以获取本机 IP，不会实际发送数据
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        print(f"服务已在网络中暴露，可通过以下地址访问:")
        print(f"**IP 地址: http://{local_ip}:8000**")
        print(f"**本地回环: http://127.0.0.1:8000**")
    except Exception as e:
        print(f"无法自动获取本机IP地址: {e}")
        print(f"请手动查找服务器的IP地址。")

    # 监听 0.0.0.0 表示可以从外部访问
    uvicorn.run(app, host="0.0.0.0", port=8000)