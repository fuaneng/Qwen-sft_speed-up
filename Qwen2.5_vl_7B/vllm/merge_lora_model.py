# merge_lora_model.py
import torch
from peft import PeftModel
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# --- 1. 配置路径 ---
# 你的 Qwen2.5-VL-7B 基础模型路径
BASE_MODEL_PATH = r"/DATA/z60046962/Qwen2.5-VL-7B-Instruct"
# 你训练好的 LoRA 权重路径 (包含 adapter_config.json 的目录)
LORA_ADAPTER_PATH = r"/DATA/z60046962/Qwen2_5_SFT-main/train/outputr/checkpoint-8400"
# 合并后新模型的保存路径
MERGED_MODEL_SAVE_PATH = r"/DATA/z60046962/Qwen2.5-VL-7B-Instruct-merged"

# --- 2. 加载模型和适配器 ---
print("正在加载基础模型...")
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    BASE_MODEL_PATH,
    torch_dtype=torch.bfloat16,
    device_map="cpu",  # 在 CPU 上加载以节省显存
)

print("正在加载 LoRA 适配器...")
model = PeftModel.from_pretrained(model, LORA_ADAPTER_PATH)

# --- 3. 合并并保存 ---
print("正在合并模型...")
# 使用 PEFT 的 merge_and_unload() 方法进行合并
model = model.merge_and_unload()
print("合并完成！")

print(f"正在将合并后的模型和处理器保存到: {MERGED_MODEL_SAVE_PATH}")
# 保存模型
model.save_pretrained(MERGED_MODEL_SAVE_PATH, max_shard_size="4GB")

# 加载并保存对应的处理器，确保新模型文件夹是完整的
processor = AutoProcessor.from_pretrained(BASE_MODEL_PATH)
processor.save_pretrained(MERGED_MODEL_SAVE_PATH)

print("所有操作完成！")