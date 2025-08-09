import torch
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from peft import PeftModel, LoraConfig, TaskType
import os

# --- 1. 配置你的模型路径 ---
base_model_path = r"/DATA/z60046962/Qwen2_2B_SFT-main/model/Qwen2-VL-2B-Instruct"
lora_path = r"/DATA/z60046962/Qwen2_2B_SFT-main/model/lora/checkpoint-26100"
merged_model_save_path = r"/DATA/z60046962/Qwen2_2B_SFT-main/model/Qwen2-VL-2B-Instruct-merged"

# --- 2. LoRA 配置 ---
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=64,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
)

print("正在加载基础模型到 GPU...")
# 【核心改动】使用 device_map="auto" 将模型加载到 GPU
model = Qwen2VLForConditionalGeneration.from_pretrained(
    base_model_path,
    torch_dtype=torch.float32, # 使用 float32 以保证合并精度
    device_map="auto" # 自动分配设备（GPU 或 CPU），cuda, cpu 等
)

print("正在加载 LoRA 适配器...")
# PeftModel 会自动将适配器加载到模型所在的设备上
model = PeftModel.from_pretrained(model, model_id=lora_path, config=config)

print("正在 GPU 上合并 LoRA 权重...")
# 合并权重并卸载适配器
model = model.merge_and_unload()
print("权重合并完成！")

# 为了保存，需要先将模型移回 CPU，这是 save_pretrained 的标准做法
print(f"正在将合并后的模型和处理器保存到: {merged_model_save_path}")
model.save_pretrained(merged_model_save_path)

# 加载并保存对应的处理器/分词器
processor = AutoProcessor.from_pretrained(base_model_path)
processor.save_pretrained(merged_model_save_path)

print("所有操作完成！合并后的模型已可用于 vLLM。")