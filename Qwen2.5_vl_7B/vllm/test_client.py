# -*- coding: utf-8 -*-
# test_client.py

import requests
import base64
import os

# --- 配置 ---
# 服务地址，如果你的服务部署在其他机器，请修改 IP 地址
API_URL = "http://127.0.0.1:8000/predict"

# --- 测试用例 ---
# 1. 使用网络图片 URL
TEST_IMAGE_URL = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
TEST_PROMPT_URL = "这张图里有什么"

# 2. 使用本地图片
# 【重要】请将 'path/to/your/local/image.jpg' 替换为一个真实存在的本地图片路径
LOCAL_IMAGE_PATH = 'path/to/your/local/image.jpg'
TEST_PROMPT_LOCAL = "详细描述一下这张图片"


def test_with_image_url(url, prompt):
    """使用图片URL测试API"""
    print("--- 正在使用图片 URL 进行测试 ---")
    payload = {
        "prompt": prompt,
        "image_url": url
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=60) # 增加超时时间
        # 检查响应状态码
        if response.status_code == 200:
            print("请求成功!")
            print("模型返回结果:")
            print(response.json()['result'])
        else:
            print(f"请求失败，状态码: {response.status_code}")
            print("错误详情:")
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"请求过程中发生错误: {e}")
    print("-" * 30 + "\n")


def test_with_local_image(image_path, prompt):
    """使用本地图片（Base64编码）测试API"""
    print("--- 正在使用本地图片进行测试 ---")

    # 检查本地文件是否存在
    if not os.path.exists(image_path):
        print(f"错误: 本地图片路径不存在 -> '{image_path}'")
        print("请修改 LOCAL_IMAGE_PATH 为一个有效的图片文件路径后再试。")
        print("-" * 30 + "\n")
        return

    # 读取图片并进行Base64编码
    try:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    except Exception as e:
        print(f"读取或编码本地图片时出错: {e}")
        print("-" * 30 + "\n")
        return

    payload = {
        "prompt": prompt,
        "image_base64": encoded_string
    }
    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        if response.status_code == 200:
            print("请求成功!")
            print("模型返回结果:")
            print(response.json()['result'])
        else:
            print(f"请求失败，状态码: {response.status_code}")
            print("错误详情:")
            print(response.text)
    except requests.exceptions.RequestException as e:
        print(f"请求过程中发生错误: {e}")
    print("-" * 30 + "\n")


if __name__ == "__main__":
    # 运行第一个测试
    test_with_image_url(TEST_IMAGE_URL, TEST_PROMPT_URL)

    # 运行第二个测试
    test_with_local_image(LOCAL_IMAGE_PATH, TEST_PROMPT_LOCAL)