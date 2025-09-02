# === 修正版：文件开头的 imports ===
import os
import json
import requests
from io import BytesIO
from PIL import Image

import torch
from datasets import Dataset

from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    AutoProcessor,
    AutoTokenizer,
    # 正确的模型类（对应 Qwen2.5-VL checkpoint）
    Qwen2_5_VLForConditionalGeneration,
)

from qwen_vl_utils import process_vision_info

# === 模型与处理器加载（确保路径与checkpoint一致） ===
model_name_or_path = "/DATA/z60046962/Qwen2.5-VL-7B-Instruct"  # 请确认路径正确

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)

# 注意：如果你的GPU/机器不支持 bfloat16，可以改为 float16 或删除 dtype
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_name_or_path,
    device_map="auto",
    dtype=torch.bfloat16,       # 若报错可改为 torch.float16 或移除
    trust_remote_code=True,
)

model.enable_input_require_grads()  # 你原来有这行，保留以支持梯度检查点

# === data preprocess 函数（保留并修正若干细节） ===
def process_func(example):
    MAX_LENGTH = 8192

    # 解析会话
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]

    # 如果 conversations 中包含 vision id（或 URL），从 template 中提取
    # 你原来用 split 的方法可以，但要防护异常
    try:
        # 取出 <|vision_start|> 和 <|vision_end|> 之间的内容（可能是 URL 或相对路径）
        file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0].strip()
    except Exception as e:
        print("无法解析 vision id/path：", e)
        return None

    # 如果是 URL，直接下载；否则拼接 base_dir 读取本地文件
    if file_path.startswith("http://") or file_path.startswith("https://"):
        try:
            resp = requests.get(file_path, timeout=10)
            resp.raise_for_status()
            image = Image.open(BytesIO(resp.content)).convert("RGB")
        except Exception as e:
            print(f"下载/打开远程图片失败：{file_path}，错误：{e}")
            return None
    else:
        base_dir = '/home/ma-user/work/f60055380/Qwen2.5_vl_7B/'  # 据实修改根据据实际情况修改=json文件中图片路径
        full_path = os.path.join(base_dir, file_path)
        try:
            image = Image.open(full_path).convert("RGB")
        except Exception as e:
            print(f"打开本地图片失败：{full_path}，错误：{e}")
            return None

    # 构造 messages，复用你的 processor/process_vision_info
    input_text = str(input_content.split("<|vision_start|>")[0])
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": f"{input_text}"}]}
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)

    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    # 转成 python 原生类型以便 map 后拼接（你原来的逻辑）
    instruction = {k: v.tolist() for k, v in inputs.items()}

    response = tokenizer(f"{output_content}", add_special_tokens=False)
    input_ids = instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    labels = ([-100] * len(instruction["input_ids"][0])) + response["input_ids"] + [tokenizer.pad_token_id]

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    # 转回 tensor
    try:
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
            "pixel_values": torch.tensor(inputs['pixel_values']),
            "image_grid_thw": torch.tensor(inputs['image_grid_thw']).squeeze(0),
        }
    except Exception as e:
        print("转换 tensor 失败：", e)
        return None

# === 读取并构建 Dataset，map -> filter（移除 None 样本） ===
train_json = r"/DATA/f60055380/Meixue_PJ/data/set/xlsx/ceshigpu_train_data_train.json"
train_ds = Dataset.from_json(train_json)

# 注意：datasets.map 在函数返回 None 时有时会出错（取决于 transformers/datasets 版本）。
# 更稳妥的办法是先使用 map，若 map 可能返回 None，可使用 remove_columns / filter。
mapped = train_ds.map(process_func, remove_columns=train_ds.column_names)  # 如果 process_func 返回 dict，会替换行内容
# 如果 process_func 可能返回 None，则使用 filter 跳过：
filtered = mapped.filter(lambda x: x is not None)  # 保留非 None
train_dataset = filtered

# === LoRA / Trainer（和你原来基本一致） ===
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,
    r=64,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
)

peft_model = get_peft_model(model, config)

args = TrainingArguments(
    output_dir=r"/DATA/f60055380/Meixue_PJ/model/output",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    logging_steps=10,
    logging_first_step=1,
    num_train_epochs=10,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)

trainer.train()
