import requests 
import json 
import time 

def test_file_upload_api(api_url, file_path, data): 
    """ 
    调用文件上传 API 的函数。 

    Args: 
        api_url (str): API 接口的完整 URL。 
        file_path (str): 本地待上传文件的路径。 
        data (dict): 其他表单数据。 
    """ 
    print(f"正在上传文件 '{file_path}' 并发送请求...") 
    try: 
        start_time = time.time() 
        
        # 准备文件和表单数据 
        files = {'file': (file_path, open(file_path, 'rb'))} 
        
        # 使用 requests.post 发送 POST 请求，注意是 files 和 data 参数 
        response = requests.post(api_url, files=files, data=data) 
        
        end_time = time.time() 
        
        # 检查响应状态码 
        if response.status_code == 200: 
            print("✅ 请求成功！") 
            response_data = response.json() 
            print(f"响应耗时: {end_time - start_time:.2f} 秒") 
            print("-" * 50) 
            print("API 响应内容:") 
            print(json.dumps(response_data, indent=2, ensure_ascii=False)) 
            print("-" * 50) 
        else: 
            print(f"❌ 请求失败，状态码: {response.status_code}") 
            print(f"错误详情: {response.text}") 
            
    except requests.exceptions.RequestException as e: 
        print(f"❌ 请求发生异常: {e}") 

if __name__ == "__main__": 
    api_endpoint = "http://10.154.39.57:8000/summarize-document/" 
    
    # 待上传文件的本地路径 
    local_file_path = r"D:\work\tain\Qwen3_8B_TL\data\xlsx\问题集1200-50.xlsx"   # 本地路径 
    
    # 其他表单数据 
    request_data = { 
        "question": "总结不同维度不同分数的规律。", 
        "llm_model_path": "/DATA/f60055380/Qwen3_8B_TL/Model/qwen3-8b-hf", 
        "embedding_model_path": "/DATA/f60055380/Qwen3_8B_TL/Model/Qwen3-Embedding-8B", 
        "max_model_len": 40960, 
        "enable_thinking": True,    #Faske 
        "chunk_size_chars": 1000, 
        "sheet_name": "Sheet1" 
    } 

    test_file_upload_api(api_endpoint, local_file_path, request_data)

