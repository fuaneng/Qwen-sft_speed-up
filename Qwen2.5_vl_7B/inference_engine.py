# inference_engine.py
import os
import tempfile
import torch
import gc
import requests
from PIL import Image
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from peft import PeftModel, LoraConfig, TaskType
import logging
from qwen_vl_utils import process_vision_info

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# 全局变量，用于存储模型和处理器
model = None
processor = None

# LoRA 配置
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=True,
    r=64,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
)

def load_model():
    """
    加载 Qwen2.5-VL 模型和处理器。
    """
    global model, processor
    if model is None or processor is None:
        logging.info("正在加载模型和处理器...")
        try:
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                r"/DATA/z60046962/Qwen2.5-VL-7B-Instruct",
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
            model = PeftModel.from_pretrained(model, model_id=r"/DATA/z60046962/Qwen2_5_SFT-main/train/outputr/checkpoint-8400", config=config)
            processor = AutoProcessor.from_pretrained(r"/DATA/z60046962/Qwen2.5-VL-7B-Instruct")
            model.eval()
            logging.info("模型和处理器加载完成。")
        except Exception as e:
            logging.error(f"模型和处理器加载失败: {e}", exc_info=True)
            model = None
            processor = None
            raise RuntimeError(f"模型初始化失败: {e}")

def resize_image(image_path, max_dim=512):
    """
    缩放图片到指定最大边长，保持纵横比，减少 GPU 内存占用。
    """
    try:
        img = Image.open(image_path).convert("RGB")
        width, height = img.size
        if max(width, height) > max_dim:
            if width > height:
                new_width = max_dim
                new_height = int(height * (max_dim / width))
            else:
                new_height = max_dim
                new_width = int(width * (max_dim / height))
            img = img.resize((new_width, new_height), Image.LANCZOS)
            logging.info(f"图片已缩放至: {img.size}")
        return img
    except Exception as e:
        logging.error(f"图片缩放失败: {e}")
        return None

def generate_response(messages: list) -> str:
    """
    根据给定的消息列表生成模型响应。
    """
    global model, processor
    if model is None or processor is None:
        raise RuntimeError("模型或处理器未加载。请确保在调用 generate_response 前已调用 load_model。")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    temp_local_image_paths = []
    generated_text = ""

    try:
        for msg in messages:
            if msg.get("role") == "user":
                for content_block in msg.get("content", []):
                    if content_block.get("type") == "image":
                        image_source = content_block.get("image")
                        if not image_source or not isinstance(image_source, str):
                            raise ValueError("图片类型的内容必须提供有效的 URL 字符串。")
                        
                        if image_source.startswith(("http://", "https://")):
                            logging.info(f"正在下载图片: {image_source}")
                            response = requests.get(image_source, stream=True, timeout=10)
                            response.raise_for_status()
                            
                            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_image_file:
                                for chunk in response.iter_content(chunk_size=8192):
                                    temp_image_file.write(chunk)
                                temp_local_image_path = temp_image_file.name
                            temp_local_image_paths.append(temp_local_image_path)

                            processed_image = resize_image(temp_local_image_path, max_dim=512)
                            if processed_image is None:
                                raise ValueError(f"图片处理失败: {image_source}")
                            
                            content_block["image"] = processed_image
                        else:
                            raise ValueError(f"不支持的图片源格式: {image_source}")
                    
                    elif content_block.get("type") == "text":
                        original_text = content_block.get('text', '')
                        content_block["text"] = f"评价这张图片:{original_text}"
                        logging.info(f"已添加触发词前缀，问题变为: {content_block['text']}")

        text_for_processor = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)

        inputs = processor(
            text=[text_for_processor],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(model.device)

        with torch.cuda.amp.autocast():
            generated_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        generated_text = output_text[0].strip() if output_text else ""
        logging.info("模型输出: " + generated_text)

        del inputs, generated_ids, generated_ids_trimmed, output_text
        for msg in messages:
            for content_block in msg.get("content", []):
                if content_block.get("type") == "image" and isinstance(content_block.get("image"), Image.Image):
                    content_block["image"].close()
                    del content_block["image"]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    except requests.exceptions.RequestException as e:
        logging.error(f"图片下载失败: {e}", exc_info=True)
        raise ValueError(f"图片下载失败: {e}")
    except RuntimeError as e:
        if "CUDA out of memory" in str(e):
            logging.error(f"CUDA 内存不足: {e}", exc_info=True)
            raise RuntimeError(f"推理失败: CUDA 内存不足")
        else:
            logging.error(f"模型推理时发生运行时错误: {e}", exc_info=True)
            raise RuntimeError(f"推理失败: {e}")
    except Exception as e:
        logging.error(f"推理过程中发生未知错误: {e}", exc_info=True)
        raise RuntimeError(f"推理失败: {e}")
    finally:
        for path in temp_local_image_paths:
            if os.path.exists(path):
                try:
                    os.remove(path)
                    logging.info(f"已删除临时图片文件: {path}")
                except OSError as e:
                    logging.error(f"删除临时文件失败 {path}: {e}")

    return generated_text