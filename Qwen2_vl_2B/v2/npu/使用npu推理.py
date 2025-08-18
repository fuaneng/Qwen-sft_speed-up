import pandas as pd
import requests
import os
import base64
import gc
import torch
import io
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForVision2Seq

# --- 1. 全局配置 ---
# 【重要】使用你刚刚合并好的模型路径
MODEL_PATH = r"/data/data/f60055380/train/ckpt/Qwen2-VL-2B-Instruct-merged-26100"

# 输入和输出文件路径
EXCEL_FILE_PATH = r"/home/ma-user/work/f60055380/data/打榜_superclu模拟_2_测试集_豆包_updated.xlsx"
SHEET_NAME = "豆包"
OUTPUT_EXCEL_FILE_PATH = r"/home/ma-user/work/f60055380/data/推理结果/打榜_superclu模拟_2_测试集_豆包_updated.xlsx"

# 推理参数
BATCH_SIZE = 10      # 批次大小
SAVE_EVERY_N_BATCH = 2 # 每处理2个批次就保存一次
MAX_TOKENS = 2048    # 模型生成文本的最大长度

# --- 2. 初始化模型和处理器 (NPU) ---
print("正在加载模型和处理器 (NPU)...")
# 检查 NPU 是否可用
if not torch.npu.is_available():
    print("NPU 设备不可用。请检查你的环境配置。")
    exit()

# 指定 NPU 设备
device = torch.device("npu:0")
print(f"正在使用设备: {device}")

# 使用 AutoModelForVision2Seq 加载模型
model = AutoModelForVision2Seq.from_pretrained(
    MODEL_PATH,
    torch_dtype=torch.bfloat16, 
    trust_remote_code=True
).to(device).eval() # 添加 .eval() 以切换到推理模式

# 加载处理器
processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
print("模型和处理器加载完成。")

# --- 3. 辅助函数 ---
def resize_image(image_bytes, max_dim=512):
    """从字节流缩放图片，返回缩放后的 PIL Image 对象"""
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
        return img
    except Exception as e:
        print(f"图片缩放失败: {e}")
        return None

# --- 4. 主执行逻辑 ---
if __name__ == "__main__":
    # 加载数据
    try:
        df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_NAME)
        required_columns = ["路径", "问题", "序号"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"错误：Excel 表格中缺少必需的列 '{col}'。")
        print(f"成功读取Excel，共 {len(df)} 行数据。")
    except Exception as e:
        print(f"读取Excel文件时发生错误: {e}")
        exit()

    if '生成' not in df.columns:
        df['生成'] = ""

    batch_data = []
    batch_indices = []
    batch_count = 0

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理数据中"):
        if pd.notna(row.get('生成')) and row.get('生成') != "":
            continue

        image_path = str(row["路径"]).strip()
        question_text = str(row["问题"]).strip()
        pil_image = None
        generated_text = ""

        if os.path.exists(image_path):
            try:
                with open(image_path, "rb") as f:
                    image_bytes = f.read()
                pil_image = resize_image(image_bytes)
                if not pil_image:
                    generated_text = "本地图片处理失败，跳过推理"
            except Exception as e:
                generated_text = f"读取本地图片文件失败: {e}"
        else:
            generated_text = "无效路径，跳过推理"
        
        if pil_image:
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"评价这张图片:{question_text}"}]}]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
            batch_data.append({'image': pil_image, 'prompt': prompt})
            batch_indices.append(idx)
        else:
            df.at[idx, '生成'] = generated_text

        if len(batch_data) == BATCH_SIZE or (idx == len(df) - 1 and batch_data):
            print(f"\n开始推理批次 {batch_count + 1}，包含 {len(batch_data)} 条数据...")
            
            try:
                images_batch = [item['image'] for item in batch_data]
                prompts_batch = [item['prompt'] for item in batch_data]

                inputs = processor(text=prompts_batch, images=images_batch, return_tensors="pt", padding=True).to(device)

                # --- 【本次修改点 1】---
                # 记录下输入 token 的长度
                input_ids_len = inputs['input_ids'].shape[1]

                with torch.no_grad():
                    generation_output = model.generate(**inputs, max_new_tokens=MAX_TOKENS, do_sample=False)
                
                # --- 【本次修改点 2】---
                # 从完整的输出 token 中，只截取新生成的部分
                new_tokens = generation_output[:, input_ids_len:]
                
                # --- 【本次修改点 3】---
                # 只对新生成的部分进行解码
                decoded_texts = processor.batch_decode(new_tokens, skip_special_tokens=True)

                # 将纯净的结果写回 DataFrame
                for i, result_text in enumerate(decoded_texts):
                    original_df_index = batch_indices[i]
                    df.at[original_df_index, '生成'] = result_text.strip()

                print(f"批次 {batch_count + 1} 推理完成。")

            except Exception as e:
                print(f"批次 {batch_count + 1} 推理失败: {e}")
                for i in range(len(batch_data)):
                    original_df_index = batch_indices[i]
                    df.at[original_df_index, '生成'] = f"批处理推理失败: {e}"

            batch_data = []
            batch_indices = []
            batch_count += 1
            
            if batch_count % SAVE_EVERY_N_BATCH == 0:
                temp_path = f"{OUTPUT_EXCEL_FILE_PATH.rsplit('.', 1)[0]}_batch_{batch_count}.xlsx"
                df.to_excel(temp_path, index=False)
                print(f"中间结果已保存至: {temp_path}")

    df.to_excel(OUTPUT_EXCEL_FILE_PATH, index=False)
    print(f"\n所有数据处理完毕！最终结果已保存至: {OUTPUT_EXCEL_FILE_PATH}")

    del model, processor
    gc.collect()
    if torch.npu.is_available():
        torch.npu.empty_cache()