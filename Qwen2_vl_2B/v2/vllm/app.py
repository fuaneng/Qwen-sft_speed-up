# -*- coding: utf-8 -*-
# @Time    : 2024/8/11 15:30
# @Author  : 编码助手
# @File    : app.py
# @Description: 使用Flask将模型推理能力部署为Web API服务。

from flask import Flask, request, jsonify
from inference_engine_llm import InferenceEngine
import threading

# --- 1. 全局配置 ---
# 【重要】使用你模型的路径
MODEL_PATH = r"/DATA/z60046962/Qwen2_2B_SFT-main/model/Qwen2-VL-2B-Instruct-merged"
MAX_TOKENS = 2048

# --- 2. 初始化Flask应用和推理引擎 ---
app = Flask(__name__)
engine = None

def initialize_engine():
    """在后台线程中初始化推理引擎以避免阻塞。"""
    global engine
    print("正在初始化推理引擎，请稍候...")
    try:
        engine = InferenceEngine(
            model_path=MODEL_PATH,
            max_tokens=MAX_TOKENS
        )
        print("推理引擎初始化成功，服务已就绪。")
    except Exception as e:
        print(f"严重错误：无法初始化推理引擎: {e}")
        # 如果引擎初始化失败，服务将无法工作。
        engine = None

@app.route('/predict', methods=['POST'])
def predict():
    """
    API端点，用于接收图文数据并返回模型生成的结果。
    请求体应为JSON格式，包含 'url' 和 'question' 字段。
    """
    if not engine:
        return jsonify({"error": "推理引擎正在初始化或初始化失败，请稍后再试。"}), 503

    if not request.json or 'url' not in request.json or 'question' not in request.json:
        return jsonify({"error": "请求体必须是包含 'url' 和 'question' 的JSON。"}), 400

    try:
        data = request.get_json()
        url = data['url']
        question = data['question']

        # InferenceEngine的process_batch方法接收一个列表
        # 因此我们将单个请求包装成一个元素的列表
        batch_data = [{'url': url, 'question': question}]
        
        print(f"收到请求: url='{url[:50]}...', question='{question}'")
        
        results = engine.process_batch(batch_data)

        # 返回列表中的第一个（也是唯一一个）结果
        response_data = {
            "generated_text": results[0] if results else "未能生成结果。"
        }
        return jsonify(response_data)

    except Exception as e:
        print(f"推理过程中发生错误: {e}")
        return jsonify({"error": f"服务器内部错误: {str(e)}"}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查端点，用于确认服务是否正常运行。"""
    if engine:
        return jsonify({"status": "ok", "message": "推理引擎已加载。"}), 200
    else:
        return jsonify({"status": "loading", "message": "推理引擎正在初始化。"}), 503


# --- 3. 启动应用 ---
if __name__ == '__main__':
    # 在一个单独的线程中启动引擎初始化，这样主线程可以立即启动Web服务
    # 这对于需要较长加载时间的模型来说是很好的实践
    init_thread = threading.Thread(target=initialize_engine)
    init_thread.daemon = True
    init_thread.start()
    
    # 启动Flask开发服务器
    # 注意：在生产环境中，应使用Gunicorn或uWSGI等WSGI服务器来运行。
    app.run(host='0.0.0.0', port=9999, debug=True)

