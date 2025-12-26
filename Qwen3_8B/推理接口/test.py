import requests
import json

def test_qwen3_api(prompt, host="127.0.0.1", port=5003):
    """
    测试 Qwen3-32B Flask 接口
    """
    url = f"http://{host}:{port}/generate_text"
    
    # 构造符合标准的 messages 格式
    payload = [
        {"role": "system", "content": "你是一个严谨的助手。"},
        {"role": "user", "content": prompt}
    ]
    
    headers = {
        "Content-Type": "application/json"
    }

    print(f"正在向接口发送请求: {url}")
    print(f"用户提问: {prompt}")
    print("-" * 30)

    try:
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        
        # 检查 HTTP 状态码
        if response.status_code == 200:
            result = response.json()
            # 获取返回的生成的文本内容
            generated_texts = result.get("generated_text", [])
            
            for i, text in enumerate(generated_texts):
                print(f"模型回答:\n{text}")
        else:
            print(f"请求失败，状态码: {response.status_code}")
            print(f"错误信息: {response.text}")
            
    except Exception as e:
        print(f"发生错误: {e}")

if __name__ == "__main__":
    # 这里可以修改为你想要测试的问题
    user_input = "请简要解释什么是大语言模型的注意力机制。"
    
    # 如果你的服务运行在其他服务器上，请修改 host 参数
    test_qwen3_api(user_input, host="127.0.0.1", port=5003)