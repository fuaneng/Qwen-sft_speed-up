# 你的请求脚本
import requests
import json

# API 地址，注意端口已更改为 8080
API_URL = "http://localhost:8080/v1/chat/completions"

# 你的图片 URL 和问题文本
image_url = "https://example.com/your-image.jpg"
question_text = "评价这张图片里有什么？"

# 构建请求的 JSON 数据
payload = {
    "messages": [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image_url
                },
                {
                    "type": "text",
                    "text": question_text
                }
            ]
        }
    ]
}

# 发送 POST 请求
try:
    headers = {"Content-Type": "application/json"}
    response = requests.post(API_URL, data=json.dumps(payload), headers=headers)
    response.raise_for_status()

    result = response.json()
    print("模型推理成功！")
    print("响应结果:", result["result"])

except requests.exceptions.RequestException as e:
    print(f"发送请求失败: {e}")
    if 'response' in locals() and response:
        print("API 响应:", response.text)
except Exception as e:
    print(f"处理响应时发生错误: {e}")