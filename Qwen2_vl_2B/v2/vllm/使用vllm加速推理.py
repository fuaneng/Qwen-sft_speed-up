# -*- coding: utf-8 -*-
import pandas as pd
import requests
import os
import base64
import gc
import torch
import io
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info

# --- 1. 全局配置 ---
# 【重要】使用你刚刚合并好的模型路径
MODEL_PATH = r"/DATA/z60046962/Qwen2_2B_SFT-main/model/Qwen2-VL-2B-Instruct-merged"

# 输入和输出文件路径
EXCEL_FILE_PATH = r"/DATA/z60046962/Qwen2_2B_SFT-main/data/测试自拓展.xlsx"
SHEET_NAME = "Sheet1"
OUTPUT_EXCEL_FILE_PATH = r"/DATA/z60046962/Qwen2_2B_SFT-main/data/推理结果_vllm_测试自拓展.xlsx"

# 推理参数
BATCH_SIZE = 8  # vLLM可以处理更大的批次，根据你的显存调整 v100,设置为10
SAVE_EVERY_N_BATCH = 5 # 每处理5个批次就保存一次，防止意外
MAX_TOKENS = 2048 # 模型生成文本的最大长度

# --- 2. 初始化vLLM和处理器 ---
print("正在加载模型和处理器 (vLLM)...")
# 配置 vLLM
llm = LLM(
    model=MODEL_PATH,
    tensor_parallel_size=1,  # 如果你有多张GPU，可以设置为多卡并行
    trust_remote_code=True,
    dtype="bfloat16",  # 或 "float16"，根据你的GPU支持情况选择
    enforce_eager=True, # 多模态模型通常需要 eager 模式
    max_model_len=8192, # 根据模型能力设定最大长度
    gpu_memory_utilization=0.8, # 占用显存比例默认值0.9,一般设置为0.7-0.8较为合理,根据显存大小调整
    # max_num_seqs=1,  # 每次推理处理的最大序列数
    # max_num_batched_seqs=1,  # 每次批处理的最大序列数       
    # max_num_gpu_blocks=1,  # 每次推理处理的最大GPU块数
    # max_num_cpu_blocks=1,  # 每次推理处理的最大CPU块数                      
)

# 配置采样参数
sampling_params = SamplingParams(
    temperature=0.0,
    top_p=1.0, # 如果temperature为0，top_p通常设为1.0
    max_tokens=MAX_TOKENS,
)

# 加载处理器
processor = AutoProcessor.from_pretrained(MODEL_PATH)
print("模型和处理器加载完成。")


# --- 3. 辅助函数 ---
def resize_image(image_bytes, max_dim=512):
    """从字节流缩放图片，返回缩放后的字节流"""
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
        
        # 将缩放后的图片存回字节流
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return buffered.getvalue()
    except Exception as e:
        print(f"图片缩放失败: {e}")
        return None

def prepare_vllm_input(prompt, image_base64):
    """为vLLM准备单条图文输入"""
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": f"data:image/png;base64,{image_base64}"},
                {"type": "text", "text": f"评价这张图片:{prompt}"}
            ],
        }
    ]
    
    # 应用聊天模板
    text_prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # 提取视觉信息
    image_inputs, _ = process_vision_info(messages)

    # 构建 vLLM 输入字典
    llm_inputs = {
        "prompt": text_prompt,
        "multi_modal_data": {"image": image_inputs},
    }
    return llm_inputs

# --- 4. 主执行逻辑 ---
if __name__ == "__main__":
    # 加载数据
    try:
        df = pd.read_excel(EXCEL_FILE_PATH, sheet_name=SHEET_NAME)
        # 确保必需的列存在
        required_columns = ["url", "问题", "序号"]
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"错误：Excel 表格中缺少必需的列 '{col}'。")
        print(f"成功读取Excel，共 {len(df)} 行数据。")
    except Exception as e:
        print(f"读取Excel文件时发生错误: {e}")
        exit()

    # 添加新列用于存放结果
    df['生成'] = ""

    input_batchlist = []
    batch_count = 0

    # 使用tqdm显示进度条
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="处理数据中"):
        image_url = str(row["url"]).strip()
        question_text = str(row["问题"]).strip()
        
        generated_text = ""
        image_base64 = ""

        if image_url.startswith(("http://", "https://")):
            try:
                # 下载图片到内存
                response = requests.get(image_url, timeout=10)
                response.raise_for_status()
                image_bytes = response.content
                
                # 缩放图片
                resized_bytes = resize_image(image_bytes)
                if resized_bytes:
                    # Base64 编码
                    image_base64 = base64.b64encode(resized_bytes).decode('utf-8')
                else:
                    generated_text = "图片处理失败，跳过推理"

            except requests.exceptions.RequestException as e:
                generated_text = f"下载图片失败: {e}"
                print(f"序号 {row['序号']}: {generated_text}")
        else:
            generated_text = "无效URL，跳过推理"

        # 如果图片处理成功，准备vLLM输入
        if image_base64:
            llm_input = prepare_vllm_input(question_text, image_base64)
            # 保存DataFrame的索引和输入数据
            input_batchlist.append((idx, llm_input))
        # 如果图片处理失败，直接记录错误信息
        else:
            df.at[idx, '生成'] = generated_text

        # 当累积满一个批次，或这是最后一条数据时，开始推理
        if len(input_batchlist) == BATCH_SIZE or (idx == len(df) - 1 and input_batchlist):
            indices, inputs = zip(*input_batchlist)
            
            print(f"\n开始推理批次 {batch_count + 1}，包含 {len(inputs)} 条数据...")
            outputs = llm.generate(list(inputs), sampling_params=sampling_params)
            
            # 将结果写回DataFrame
            for i, output in enumerate(outputs):
                original_df_index = indices[i]
                result_text = output.outputs[0].text.strip()
                df.at[original_df_index, '生成'] = result_text

            print(f"批次 {batch_count + 1} 推理完成。")
            
            # 清空批次列表并增加计数器
            input_batchlist = []
            batch_count += 1
            
            # 定期保存
            if batch_count % SAVE_EVERY_N_BATCH == 0:
                temp_path = f"{OUTPUT_EXCEL_FILE_PATH.rsplit('.', 1)[0]}_batch_{batch_count}.xlsx"
                df.to_excel(temp_path, index=False)
                print(f"中间结果已保存至: {temp_path}")

    # 保存最终的完整结果
    df.to_excel(OUTPUT_EXCEL_FILE_PATH, index=False)
    print(f"\n所有数据处理完毕！最终结果已保存至: {OUTPUT_EXCEL_FILE_PATH}")

    # 清理资源 (可选)
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()