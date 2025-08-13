import torch
from datasets import Dataset
from modelscope import snapshot_download, AutoTokenizer
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    AutoProcessor,
    AutoModelForVision2Seq, # <-- 【MODIFICATION 1】: Import the correct AutoClass for Vision-to-Sequence models
)
import json
import requests
from PIL import Image
from io import BytesIO

# ... your process_func and predict functions remain the same ...
def process_func(example):
    MAX_LENGTH = 8192
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]
    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]

    if file_path.startswith("http://") or file_path.startswith("https://"):
        try:
            response = requests.get(file_path)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading image from {file_path}: {e}")
            return None
        except IOError as e:
            print(f"Error opening image from {file_path}: {e}")
            return None
    else:
        try:
            image = Image.open(file_path).convert("RGB")
        except IOError as e:
            print(f"Error opening local image from {file_path}: {e}")
            return None

    input_text = str(input_content.split("<|vision_start|>")[0])
    messages = [
        {"role": "user", "content": [{"type": "image", "image": image, "resized_height": 512, "resized_width": 512,}, {"type": "text", "text": f"{input_text}"},],}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    inputs = {key: value.tolist() for key, value in inputs.items()}
    instruction = inputs

    response = tokenizer(f"{output_content}", add_special_tokens=False)
    input_ids = instruction["input_ids"][0] + response["input_ids"] + [tokenizer.pad_token_id]
    attention_mask = instruction["attention_mask"][0] + response["attention_mask"] + [1]
    labels = ([-100] * len(instruction["input_ids"][0]) + response["input_ids"] + [tokenizer.pad_token_id])

    if len(input_ids) > MAX_LENGTH:
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values'])
    inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels, "pixel_values": inputs['pixel_values'], "image_grid_thw": inputs['image_grid_thw']}

def predict(messages, model):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return output_text[0]
# ==================== Main Flow ====================

tokenizer = AutoTokenizer.from_pretrained("/DATA/z60046962/Qwen2.5-VL-7B-Instruct", use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained("/DATA/z60046962/Qwen2.5-VL-7B-Instruct", trust_remote_code=True)

# 【MODIFICATION 2】: Use AutoModelForVision2Seq to load the model
# This class is designed for multimodal models like Qwen-VL.
model = AutoModelForVision2Seq.from_pretrained(
    "/DATA/z60046962/Qwen2.5-VL-7B-Instruct",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.enable_input_require_grads()

# ... rest of your code remains the same ...
train_json_path = r"/DATA/z60046962/Qwen2_5_SFT-main/train/caption/data_self_5944.json"
with open(train_json_path, 'r') as f:
    data = json.load(f)
    train_data = data[:-4]
    test_data = data[-4:]

train_ds_path = r"/DATA/z60046962/Qwen2_5_SFT-main/train/caption/data_self_5944_train.json"
test_ds_path = r"/DATA/z60046962/Qwen2_5_SFT-main/train/caption/data_self_5944_test.json"
with open(train_ds_path, "w") as f:
    json.dump(train_data, f)
with open(test_ds_path, "w") as f:
    json.dump(test_data, f)

train_ds = Dataset.from_json(train_ds_path)
train_dataset = train_ds.map(process_func).filter(lambda x: x is not None)

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
    output_dir=r"/DATA/z60046962/Qwen2_5_SFT-main/train/output0813",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    logging_steps=10,
    logging_first_step=1,
    num_train_epochs=10,
    save_steps=300,
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

if trainer.is_world_process_zero():
    print("Training finished! Model saved in the output directory.")
    final_model_path = r"/DATA/z60046962/Qwen2_5_SFT-main/train/output0813/final_model"
    peft_model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"Final LoRA adapter and tokenizer have been saved to: {final_model_path}")