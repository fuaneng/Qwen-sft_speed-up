# -*- coding: utf-8 -*-
import requests
import json
import time

# --- 1. 配置测试参数 ---
# 你的API服务地址和端口
API_HOST = "http://127.0.0.1" 
API_PORT = 9999
API_URL = f"{API_HOST}:{API_PORT}"

# 你想用于测试的图片URL和问题
TEST_DATA = [
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Two_men_eating_hot_dogs_in_a_hot_dog_stand.jpg/1200px-Two_men_eating_hot_dogs_in_a_hot_dog_stand.jpg",
        "question": "这张图片里有什么？"
    },
    {
        "url": "https://www.nasa.gov/wp-content/uploads/2023/05/hubble-m101-image-01.jpg",
        "question": "描述一下这幅图片的场景。"
    },
    {
        "url": "http://invalid-image-url.com/non_existent.jpg",
        "question": "这张图是什么？"  # 测试无效URL
    },
    {
        "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/36/Two_men_eating_hot_dogs_in_a_hot_dog_stand.jpg/1200px-Two_men_eating_hot_dogs_in_a_hot_dog_stand.jpg",
        "question": "告诉我图片里的人在干什么？"
    }
]


# --- 2. 编写测试函数 ---

def health_check():
    """测试服务的健康检查端点。"""
    print("--- 正在进行健康检查 ---")
    try:
        response = requests.get(f"{API_URL}/health")
        print(f"状态码: {response.status_code}")
        print(f"响应内容: {response.json()}")
        if response.status_code == 200:
            print("服务健康检查通过！")
            return True
        else:
            print("健康检查失败，服务可能正在初始化或发生错误。")
            return False
    except requests.exceptions.ConnectionError:
        print("无法连接到API服务，请确保服务已启动。")
        return False

def test_predict_endpoint():
    """测试 /predict API 端点。"""
    print("\n--- 正在测试 /predict 端点 ---")
    
    for i, data in enumerate(TEST_DATA):
        print(f"\n--- 测试用例 {i+1} ---")
        print(f"发送请求: url='{data['url']}', question='{data['question']}'")
        try:
            headers = {"Content-Type": "application/json"}
            response = requests.post(f"{API_URL}/predict", data=json.dumps(data), headers=headers, timeout=60)
            
            print(f"状态码: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"响应内容:\n{json.dumps(result, indent=2, ensure_ascii=False)}")
                print("生成文本长度:", len(result.get("generated_text", "")))
            else:
                print(f"错误响应: {response.json()}")
        
        except requests.exceptions.RequestException as e:
            print(f"请求失败: {e}")

def main():
    """主函数，运行所有测试。"""
    if not health_check():
        print("等待服务初始化...")
        # 等待一段时间，再重试健康检查
        time.sleep(30)  
        if not health_check():
            print("服务仍未就绪，测试终止。")
            return

    test_predict_endpoint()

if __name__ == "__main__":
    main()