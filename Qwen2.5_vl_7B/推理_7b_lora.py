import gc
import os
import tempfile

import openpyxl  # 导入 openpyxl 用于逐行写入
import pandas as pd
import requests
import torch
from PIL import Image
from peft import LoraConfig, PeftModel, TaskType
from qwen_vl_utils import process_vision_info  # 确保 qwen_vl_utils 在 Python 路径中
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# LoRA 配置
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    # 根据 Qwen2.5-VL-7B 模型结构调整 target_modules
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    inference_mode=True,
    r=64,
    lora_alpha=64,
    lora_dropout=0.05,
    bias="none",
)

# 加载模型和处理器
print("正在加载模型和处理器...")
# 更新模型路径和 torch_dtype
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    r"/DATA/z60046962/Qwen2.5-VL-7B-Instruct",  # 假设这是你的 7B 模型路径
    torch_dtype=torch.bfloat16,  # 7B 模型推荐使用 bfloat16
    device_map="auto",  # 自动分配设备（GPU 或 CPU）
    # attn_implementation="flash_attention_2", # 如果你的硬件支持，可以取消注释以启用 Flash Attention 2
)
# 请确保这里的 LoRA 模型 ID 是针对 2.5-VL-7B LoRA 微调后的检查点路径
model = PeftModel.from_pretrained(
    model,
    model_id=r"/DATA/z60046962/Qwen2_5_SFT-main/train/outputr/checkpoint-8100",
    config=config,
)
# 更新处理器路径
processor = AutoProcessor.from_pretrained(
    r"/DATA/z60046962/Qwen2.5-VL-7B-Instruct"
)  # 假设这是你的 7B 模型路径
model.eval()  # 将模型设置为评估模式，禁用 dropout 等
print("模型和处理器加载完成。")


# 函数：缩放图片以减少内存占用
def resize_image(image_path, max_dim=512):
    """
    缩放图片到指定最大边长，保持纵横比，减少 GPU 内存占用。
    返回缩放后的 PIL Image 对象，而不是保存到新文件。
    """
    try:
        img = Image.open(image_path).convert("RGB")  # 确保是RGB模式
        width, height = img.size
        if max(width, height) > max_dim:
            if width > height:
                new_width = max_dim
                new_height = int(height * (max_dim / width))
            else:
                new_height = max_dim
                new_width = int(width * (max_dim / height))
            img = img.resize((new_width, new_height), Image.LANCZOS)
            print(f"图片已缩放至: {img.size}")
        return img
    except Exception as e:
        print(f"图片缩放失败: {e}")
        return None  # 缩放失败时返回 None


# 定义 Excel 文件路径和表名
excel_file_path = (
    r"/DATA/z60046962/Qwen2_5_SFT-main/train/data_tuili/打榜_superclu模拟_2_测试集_beifen.xlsx"
)
sheet_name = "豆包"  # Excel 分表名称
output_excel_file_path = (
    r"/DATA/z60046962/Qwen2_5_SFT-main/train/data_tuili/推理结果_打榜2_豆包-8100.xlsx"
)  # 输出结果文件路径

# 定义输出 Excel 的列名
output_columns = ["序号", "url", "问题", "生成"]

# 读取 Excel 文件
try:
    df = pd.read_excel(excel_file_path, sheet_name=sheet_name)
    required_columns = ["url", "问题", "序号"]
    for col in required_columns:
        if col not in df.columns:
            print(f"错误：Excel 表格中缺少必需的列 '{col}'。请检查列名。")
            exit()
    print(
        f"成功读取 Excel 文件 '{excel_file_path}' 的工作表 '{sheet_name}'，共 {len(df)} 行数据。"
    )
except FileNotFoundError:
    print(f"错误：未找到文件 '{excel_file_path}'。请检查文件路径是否正确。")
    exit()
except KeyError:
    print(f"错误：未找到名为 '{sheet_name}' 的工作表。请检查工作表名称是否正确。")
    exit()
except Exception as e:
    print(f"读取 Excel 文件时发生错误: {e}")
    exit()

# 初始化或加载输出 Excel 文件
# 使用 openpyxl 进行逐行写入，避免一次性加载和保存整个DataFrame
if not os.path.exists(output_excel_file_path):
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "推理结果"
    sheet.append(output_columns)
    workbook.save(output_excel_file_path)
    print(f"已创建新的输出文件: '{output_excel_file_path}' 并写入表头。")
else:
    workbook = openpyxl.load_workbook(output_excel_file_path)
    sheet = workbook.active
    print(f"已加载现有输出文件: '{output_excel_file_path}'。")

# 遍历 Excel 的每一行数据
for index, row in df.iterrows():
    # 每轮循环开始时清理 CUDA 缓存和 Python 垃圾回收
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    image_url = str(row["url"]).strip()  # 图片 URL
    question_text = str(row["问题"]).strip()  # 问题文本
    sequence_id = row["序号"]  # 序号

    generated_text = ""  # 初始化本行的生成文本
    temp_local_image_path = None  # 初始化临时文件路径

    print(f"\n--- 正在处理第 {index + 1}/{len(df)} 行数据 (序号: {sequence_id}) ---")
    print(f"原始图片 URL: {image_url}")
    print(f"问题文本: {question_text}")

    # 检查 URL 是否有效（仅处理 HTTP/HTTPS 链接）
    if not image_url.startswith(("http://", "https://")):
        print(f"跳过：无效图片 URL 格式 (非 HTTP/HTTPS): {image_url}")
        generated_text = "无效URL，跳过推理"
    else:
        # 下载图片到临时文件
        try:
            response = requests.get(image_url, stream=True, timeout=10)  # 设置超时时间
            response.raise_for_status()  # 检查请求是否成功

            # 使用 tempfile.NamedTemporaryFile 创建临时文件，确保会被删除
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as temp_image_file:
                for chunk in response.iter_content(chunk_size=8192):
                    temp_image_file.write(chunk)
                temp_local_image_path = temp_image_file.name  # 保存临时文件路径

            print(f"图片已下载到临时路径: {temp_local_image_path}")

            # 缩放图片并获取 PIL Image 对象
            # resize_image 函数现在返回 PIL Image 对象
            processed_image = resize_image(temp_local_image_path, max_dim=512)

            if processed_image is None:  # 如果图片处理失败
                generated_text = "图片处理失败，跳过推理"
            else:
                # 构建消息列表，包含 PIL Image 对象和问题文本
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": processed_image},  # 直接传入 PIL Image 对象
                            {"type": "text", "text": f"评价这张图片:{question_text}"},
                        ],
                    }
                ]

                # 推理过程
                try:
                    text = processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    # `process_vision_info` 会处理 PIL Image 对象
                    image_inputs, video_inputs = process_vision_info(messages)
                    inputs = processor(
                        text=[text],
                        images=image_inputs,
                        videos=video_inputs,
                        padding=True,
                        return_tensors="pt",
                    )
                    # 将输入移到模型所在的设备
                    inputs = inputs.to(model.device)

                    # 使用混合精度推理以减少内存占用
                    with torch.cuda.amp.autocast():
                        generated_ids = model.generate(**inputs, max_new_tokens=2048)
                    generated_ids_trimmed = [
                        out_ids[len(in_ids) :]
                        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                    ]
                    output_text = processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False,
                    )
                    generated_text = output_text[0].strip() if output_text else ""
                    print("模型输出:", generated_text)

                    # 显式删除不再需要的张量
                    del inputs, generated_ids, generated_ids_trimmed, output_text, processed_image
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                except RuntimeError as e:
                    if "CUDA out of memory" in str(e):
                        print(f"CUDA 内存不足: {e}")
                        generated_text = "推理失败: CUDA 内存不足"
                    else:
                        print(f"推理图片 '{temp_local_image_path}' 时发生其他运行时错误: {e}")
                        generated_text = f"推理失败: {e}"
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()  # 尝试清理缓存
                except Exception as e:
                    print(f"推理图片 '{temp_local_image_path}' 时发生未知错误: {e}")
                    generated_text = f"推理失败: {e}"

        except requests.exceptions.RequestException as e:
            print(f"下载图片失败 '{image_url}': {e}")
            generated_text = f"图片下载失败: {e}"
        except Exception as e:
            print(f"处理图片时发生未知错误: {e}")
            generated_text = f"图片处理失败: {e}"
        finally:
            # 无论推理成功与否，尝试删除临时图片文件
            if temp_local_image_path and os.path.exists(temp_local_image_path):
                try:
                    os.remove(temp_local_image_path)
                    print(f"已删除临时图片文件: {temp_local_image_path}")
                except OSError as e:
                    print(f"删除临时文件失败 {temp_local_image_path}: {e}")

    # --- 每处理一条数据后保存结果到 Excel ---
    data_row = [sequence_id, image_url, question_text, generated_text]
    try:
        sheet.append(data_row)
        workbook.save(output_excel_file_path)  # 每次循环保存，确保数据不丢失
        print(f"第 {index + 1} 行推理结果已保存到 '{output_excel_file_path}'")
    except Exception as e:
        print(f"保存第 {index + 1} 行结果到 Excel 失败: {e}")

print("\n所有数据处理完毕。")