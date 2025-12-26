import os
import torch
import argparse
from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

# 全局变量
_model = None
_tokenizer = None

def load_model_and_tokenizer():
    """
    加载 Qwen3-32B 模型和分词器
    """
    global _model, _tokenizer
    if _model is None and _tokenizer is None:
        # 使用你第一段代码提供的模型路径
        model_path = "/DATA/f60055380/Qwen3_8B_TL/Model/Qwen3-32B"
        print(f"正在加载模型: {model_path}...")

        # 1. 加载分词器
        _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        
        # 2. 加载模型 (针对单卡 A100 优化)
        # 注意：32B 模型在 float16 下占用约 64GB 显存
        _model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto", 
            trust_remote_code=True
        )
        _model.eval()
        print("Qwen3-32B 加载完成。")

@app.route('/generate_text', methods=['POST'])
def generate_text():
    load_model_and_tokenizer()

    try:
        data = request.get_json()
    except Exception:
        return jsonify({"error": "无效的 JSON 格式"}), 400

    if data is None or not isinstance(data, list):
        return jsonify({"error": "输入数据应为消息列表 (Messages List)"}), 400

    # --- 嵌入你的示例代码逻辑：禁用思考模式 ---
    # 明确设置 enable_thinking=False
    text = _tokenizer.apply_chat_template(
        data,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False  # 这里已改为 False 以禁用思考模式
    )
    # -------------------------------------------

    # 1. 准备推理输入
    inputs = _tokenizer([text], return_tensors="pt").to("cuda")

    # 2. 执行推理
    with torch.no_grad():
        generated_ids = _model.generate(
            **inputs,
            max_new_tokens=2048, 
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=_tokenizer.eos_token_id
        )

    # 3. 剪裁输入部分，只保留生成的回答
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    
    # 4. 解码输出
    output_text = _tokenizer.batch_decode(generated_ids_trimmed, skip_special_tokens=True)
    
    return jsonify({"generated_text": output_text})

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Qwen3-32B Flask 接口服务")
    parser.add_argument('--port', type=int, default=5003, help='监听端口号 (默认: 5003)')
    args = parser.parse_args()

    app.run(host='0.0.0.0', port=args.port)