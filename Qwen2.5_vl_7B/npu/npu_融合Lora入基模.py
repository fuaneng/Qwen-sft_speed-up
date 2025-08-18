import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from peft import PeftModel, LoraConfig, TaskType
import os

# --- 1. 配置路径 ---
# 基础模型路径
BASE_MODEL_PATH = r"/DATA/z60046962/Qwen2.5-VL-7B-Instruct"
# LoRA (PEFT) 权重路径
PEFT_MODEL_PATH = r"/DATA/z60046962/Qwen2_5_SFT-main/train/outputr/checkpoint-9300"
# 【重要】指定一个新的路径，用于保存合并后的模型
MERGED_MODEL_PATH = r"/DATA/z60046962/Qwen2.5-VL-7B-Instruct-Merged"

# --- 2. 加载模型和适配器 ---
print(f"正在从 {BASE_MODEL_PATH} 加载基础模型...")
# 注意：这里加载到CPU上进行合并操作，避免占用NPU/GPU
base_model = AutoModelForVision2Seq.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    device_map="cpu" 
)

print(f"正在从 {PEFT_MODEL_PATH} 加载 LoRA 适配器...")
# LoRA 配置需要和训练时一致
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=True,
    r=64,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
)
model = PeftModel.from_pretrained(base_model, model_id=PEFT_MODEL_PATH, config=config)
print("LoRA 适配器加载完成。")


# --- 3. 合并权重并卸载适配器 ---
print("正在合并 LoRA 权重...")
# merge_and_unload() 会将LoRA权重合并到基础模型中，并返回一个新的、不含peft结构的模型
model = model.merge_and_unload()
print("权重合并完成。")

# --- 4. 保存合并后的模型和处理器 ---
print(f"正在将合并后的模型保存到: {MERGED_MODEL_PATH}")
# 创建目标目录
os.makedirs(MERGED_MODEL_PATH, exist_ok=True)
# 保存模型
model.save_pretrained(MERGED_MODEL_PATH)

# 同时，保存对应的处理器，使其成为一个完整的模型包
processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH, trust_remote_code=True)
processor.save_pretrained(MERGED_MODEL_PATH)

print("合并后的模型和处理器已成功保存！")