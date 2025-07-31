import torch
from datasets import Dataset
from qwen_vl_utils import process_vision_info
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
)
import json

from transformers import Qwen2_5_VLForConditionalGeneration,AutoTokenizer,AutoProcessor

import requests
from PIL import Image
from io import BytesIO

def process_func(example):
    """
    将数据集进行预处理
    """
    MAX_LENGTH = 8192
    
    conversation = example["conversations"]
    input_content = conversation[0]["value"]
    output_content = conversation[1]["value"]
    file_path = input_content.split("<|vision_start|>")[1].split("<|vision_end|>")[0]  # 获取图像路径

    # Process images from URL or local path
    if file_path.startswith("http://") or file_path.startswith("https://"):
        try:
            response = requests.get(file_path)
            response.raise_for_status()  # Check if the HTTP request was successful
            image = Image.open(BytesIO(response.content)).convert("RGB")
        except requests.exceptions.RequestException as e:
            print(f"Error downloading image from {file_path}: {e}")
            return None  # Return None to skip this sample if download fails
        except IOError as e:
            print(f"Error opening image from {file_path}: {e}")
            return None  # Return None if image processing fails
    else:
        # If it's a local path, continue to load with PIL.Image.open()
        try:
            image = Image.open(file_path).convert("RGB")
        except IOError as e:
            print(f"Error opening local image from {file_path}: {e}")
            return None # Return None if local image opening fails

    # If image processing failed, return None
    if image is None:
        return None

    input_text = str(input_content.split("<|vision_start|>")[0])
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image, # Pass the PIL Image object
                    "resized_height": 280,
                    "resized_width": 280,
                },
                {"type": "text", "text": f"{input_text}"},
            ],
        }
    ]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )  # Get the text
    image_inputs, video_inputs = process_vision_info(messages)  # Get processed vision data
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {key: value.tolist() for key, value in inputs.items()}  # tensor -> list for concatenation
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
    if len(input_ids) > MAX_LENGTH:  # Truncate if too long
        input_ids = input_ids[:MAX_LENGTH]
        attention_mask = attention_mask[:MAX_LENGTH]
        labels = labels[:MAX_LENGTH]

    input_ids = torch.tensor(input_ids)
    attention_mask = torch.tensor(attention_mask)
    labels = torch.tensor(labels)
    inputs['pixel_values'] = torch.tensor(inputs['pixel_values'])
    inputs['image_grid_thw'] = torch.tensor(inputs['image_grid_thw']).squeeze(0)  # Change from (1,h,w) to (h,w)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels,
            "pixel_values": inputs['pixel_values'], "image_grid_thw": inputs['image_grid_thw']}


def predict(messages, model):
    # Prepare for inference
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

    # Generate output
    generated_ids = model.generate(**inputs, max_new_tokens=2048)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    return output_text[0]


# Use Transformers to load model weights
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    '/DATA/z60046962/Qwen2.5-VL-7B-Instruct',
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True,
)

# Load tokenizer and processor
tokenizer = AutoTokenizer.from_pretrained('/DATA/z60046962/Qwen2.5-VL-7B-Instruct',use_fast=False, trust_remote_code=True)
processor = AutoProcessor.from_pretrained('/DATA/z60046962/Qwen2.5-VL-7B-Instruct')

# Enable gradient updates
model.enable_input_require_grads()


# Process dataset: read json file
# Split into training and test sets, save as data_vl_train.json and data_vl_test.json
# >>>>>>>>>>>>>>>>>> IMPORTANT CHANGE HERE <<<<<<<<<<<<<<<<<<<<
# Replace 'your_raw_data.json' with the actual name of your JSON file containing the dataset.
train_json_path = r"/DATA/z60046962/Qwen2_5_SFT-main/data/data_maker/your_raw_data.json" 
# If your data is indeed in a file within '/DATA/z60046962/Qwen2_5_SFT-main/train/caption', 
# please specify the exact file name. For example:
# train_json_path = r"/DATA/z60046962/Qwen2_5_SFT-main/train/caption/my_dataset.json"


with open(train_json_path, 'r') as f:
    data = json.load(f)
    train_data = data[:-4]
    test_data = data[-4:]


with open("/DATA/z60046962/Qwen2_5_SFT-main/data/data_maker/data_vl_train_50.json", "w") as f:
    json.dump(train_data, f)

with open("/DATA/z60046962/Qwen2_5_SFT-main/data/data_maker/data_vl_test_50.json", "w") as f:
    json.dump(test_data, f)

# Filter out None values from process_func before creating the dataset
train_ds = Dataset.from_json("/DATA/z60046962/Qwen2_5_SFT-main/data/data_maker/data_vl_train_50.json")
train_dataset = train_ds.map(process_func).filter(lambda x: x is not None)
# Configure LoRA
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    inference_mode=False,  # Training mode
    r=64,  # Lora rank
    lora_alpha=64,  # Lora alpha, see LoRA principle for details
    lora_dropout=0.1,  # Dropout ratio
    bias="none",
)

# Get LoRA model
peft_model = get_peft_model(model, config)

# Configure training arguments
args = TrainingArguments(
    output_dir="/DATA/z60046962/Qwen2_5_SFT-main/output_dir",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    logging_steps=10,
    num_train_epochs=2,
    save_steps=100,
    learning_rate=1e-4,
    save_on_each_node=True,
    gradient_checkpointing=True,
    report_to="none",
)

# Configure Trainer
trainer = Trainer(
    model=peft_model,
    args=args,
    train_dataset=train_dataset,
    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True),
)
# Start model training
trainer.train()