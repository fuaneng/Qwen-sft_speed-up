import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from qwen_vl_utils import process_vision_info
# from swanlab.integration.transformers import SwanLabCallback
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    Qwen2VLForConditionalGeneration,
    AutoProcessor,
)
import json
import requests
from PIL import Image
from io import BytesIO


def process_func(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 8192
    input_ids, attention_mask, labels = [], [], []
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]
    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]  # 获取图像路径

    # 添加：处理 URL 图片的逻辑
    if file_path.startswith("http://") or file_path.startswith("https://"):
        try:
            response = requests.get(file_path)
            response.raise_for_status()  # 检查HTTP请求是否成功
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading image from {file_path}: {e}")
            # 如果下载失败，返回 None，这样在后续处理中可以跳过这个样本
            return None 
        except IOError as e:
            print(f"Error opening image from {file_path}: {e}")
            return None 
    else:
        # 如果是本地路径，继续使用 PIL.Image.open() 加载
        try:
            image = Image.open(file_path).convert("RGB")
        except IOError as e:
            print(f"Error opening local image from {file_path}: {e}")
            return None 

    #添加
    input_text = str(input_content.split("<|vision_start|>")[0])
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    # 传递 PIL Image 对象
                    "image": image, 
                    "resized_height": 280,
                    "resized_width": 280,
                },
                {"type": "text", "text": f"{input_text}"},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )  # 获取文本
    image_inputs, video_inputs = process_vision_info(messages)  # 获取数据数据（预处理过）
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.tolist() for key, value in inputs.items()}  # tensor -> list,为了方便拼接
    instruction = inputs

    response = tokenizer(f"{output_content}", add_special_tokens=False)

    input_ids = (
                instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    )

    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    labels = (
                [-100] * len(instruction["input_ids"][0])
                + response["input_ids"]
                + [tokenizer.pad_token_id]
    )
    if len(input_ids) > MAX_LENGTH:  # 做一个截断
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values'])
    inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)  # 由（1,h,w)变换为（h,w）
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
            "pixel_values": inputs['pixel_values'], "image_grid_thw": inputs['image_grid_thw']}


def predict(messages, model):
    # 准备推理
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = inputs.to("cuda")

    # 生成输出
    generated_ids = model.generate(**inputs, max_new_tokens=1024)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]


# 使用Transformers加载模型权重
tokenizer = AutoTokenizer.from_pretrained("/DATA/LYC/Qwen2_VL_2B_SFT-main/models/Qwen2-VL-2B-Instruct", use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained("/DATA/LYC/Qwen2_VL_2B_SFT-main/models/Qwen2-VL-2B-Instruct")

model = Qwen2VLForConditionalGeneration.from_pretrained("/DATA/LYC/Qwen2_VL_2B_SFT-main/models/Qwen2-VL-2B-Instruct", device_map="auto",torch_dtype=torch.bfloat16, trust_remote_code=True, )
model.enable_input_require_grads()  # 开启梯度检查点时，要执行该方法

# 处理数据集：读取json文件
# 拆分成训练集和测试集，保存为data_vl_train.json和data_vl_test.json
train_json_path = "/DATA/LYC/data/data_maker/data_self_all_100.json"
with open(train_json_path, 'r') as f:
    data = json.load(f)
    train_data = data[:-4]
    test_data = data[-4:]

with open("/DATA/LYC/data/data_maker/data_self_all_train_100.json", "w") as f:
    json.dump(train_data, f)

with open("/DATA/LYC/data/data_maker/data_self_all_test_100.json", "w") as f:
    json.dump(test_data, f)

train_ds = Dataset.from_json("/DATA/LYC/data/data_maker/data_self_all_train_100.json")

# 重要的修改：使用 filter() 移除 process_func 返回 None 的样本
train_dataset = train_ds.map(process_func).filter(lambda x: x is not None)

# 配置LoRA
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # 训练模式
    r=64,  # Lora 秩
    lora_alpha=64,  # Lora alaph，具体作用参见 Lora 原理
    lora_dropout=0.1,  # Dropout 比例
    bias="none",
)


# 获取LoRA模型
peft_model = get_peft_model(model, config)

# 配置训练参数
args = TrainingArguments(
    output_dir="/DATA/LYC/Qwen2_VL_2B_SFT-main/models/output_dir",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    logging_steps=10,
    logging_first_step=5,
    num_train_epochs=2,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

# 配置Trainer
trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
    #callbacks=[swanlab_callback],
)
# 开启模型训练
trainer.train()