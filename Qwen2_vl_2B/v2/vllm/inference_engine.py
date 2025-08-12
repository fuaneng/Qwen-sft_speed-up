
# @File    : inference_engine.py
# @Description: 封装了vLLM模型加载、图像处理和推理的核心逻辑。

import base64
import io
import requests
import torch
import gc
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

class InferenceEngine:
    """
    一个封装了Qwen-VL模型推理过程的引擎。
    
    这个类负责：
    1. 加载vLLM模型和处理器。
    2. 提供处理单批次图文数据的接口。
    3. 封装图片下载、缩放、编码等预处理步骤。
    """
    def __init__(self, model_path: str, max_tokens: int = 2048, gpu_memory_utilization: float = 0.7):
        """
        初始化推理引擎。

        Args:
            model_path (str): 模型的路径。
            max_tokens (int): 模型生成文本的最大长度。
            gpu_memory_utilization (float): GPU显存利用率。
        """
        print("正在初始化推理引擎...")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. 加载vLLM模型
        try:
            self.llm = LLM(
                model=model_path,
                tensor_parallel_size=1, # 如果你有多张GPU，可以设置为多卡并行
                trust_remote_code=True,
                dtype="bfloat16",
                enforce_eager=True,
                max_model_len=8192,
                gpu_memory_utilization=gpu_memory_utilization,
            )
        except Exception as e:
            print(f"加载vLLM模型失败: {e}")
            raise

        # 2. 加载处理器
        self.processor = AutoProcessor.from_pretrained(model_path)
        
        # 3. 配置采样参数
        self.sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            max_tokens=max_tokens,
        )
        print("推理引擎初始化完成。")

    def _resize_image(self, image_bytes: bytes, max_dim: int = 512) -> bytes | None:
        """
        从字节流缩放图片，返回缩放后的字节流。

        Args:
            image_bytes (bytes): 原始图片的字节流。
            max_dim (int): 图片最长边的最大尺寸。

        Returns:
            bytes | None: 缩放后图片的字节流，如果失败则返回None。
        """
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

    def _prepare_single_input(self, prompt: str, image_base64: str) -> dict:
        """
        为vLLM准备单条图文输入。

        Args:
            prompt (str): 文本提示。
            image_base64 (str): Base64编码的图片字符串。

        Returns:
            dict: 准备好的vLLM输入字典。
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": f"data:image/png;base64,{image_base64}"},
                    {"type": "text", "text": f"评价这张图片:{prompt}"}
                ],
            }
        ]
        
        text_prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        image_inputs, _ = process_vision_info(messages)

        return {
            "prompt": text_prompt,
            "multi_modal_data": {"image": image_inputs},
        }

    def process_batch(self, batch_data: list[dict]) -> list[str]:
        """
        处理一个批次的数据并返回生成结果。

        Args:
            batch_data (list[dict]): 一个批次的数据，每个字典包含 'url' 和 'question'。
                                     例如: [{'url': '...', 'question': '...'}, ...]

        Returns:
            list[str]: 包含每个输入对应的生成文本的列表。
        """
        input_batchlist = []
        results = [""] * len(batch_data)
        original_indices = [] # 记录原始批次中的索引

        for i, item in enumerate(batch_data):
            image_url = str(item.get("url", "")).strip()
            question_text = str(item.get("question", "")).strip()
            
            generated_text = ""
            image_base64 = ""

            if image_url.startswith(("http://", "https://")):
                try:
                    response = requests.get(image_url, timeout=10)
                    response.raise_for_status()
                    image_bytes = response.content
                    
                    resized_bytes = self._resize_image(image_bytes)
                    if resized_bytes:
                        image_base64 = base64.b64encode(resized_bytes).decode('utf-8')
                    else:
                        generated_text = "图片处理失败，跳过推理"
                except requests.exceptions.RequestException as e:
                    generated_text = f"下载图片失败: {e}"
            else:
                generated_text = "无效URL，跳过推理"

            if image_base64:
                llm_input = self._prepare_single_input(question_text, image_base64)
                input_batchlist.append(llm_input)
                original_indices.append(i) # 记录这个成功处理的输入的原始索引
            else:
                results[i] = generated_text # 对于处理失败的，直接记录错误信息

        # 如果有任何成功处理的输入，则进行推理
        if input_batchlist:
            outputs = self.llm.generate(input_batchlist, sampling_params=self.sampling_params)
            
            # 将推理结果放回其在原始批次中的正确位置
            for i, output in enumerate(outputs):
                original_idx = original_indices[i]
                result_text = output.outputs[0].text.strip()
                results[original_idx] = result_text
        
        return results

    def cleanup(self):
        """清理资源。"""
        print("正在清理推理引擎资源...")
        del self.llm
        del self.processor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("清理完成。")

