import pandas as pd
import json

def process_excel_to_json(excel_path, sheet_name='Sheet1', max_entries=None, output_json_path='data_self_all_100.json'):
    """
    读取Excel文件，并将其转换为指定格式的JSON文件。

    Args:
        excel_path (str): Excel文件的路径，例如 'D:/work/tain/gemini数据.xlsx'。
        sheet_name (str): Excel文件中要读取的sheet名称，默认为 'Sheet1'。
        max_entries (int, optional): 最大处理的行数。如果为None，则处理所有行。
        output_json_path (str): 输出JSON文件的路径，默认为 'data_self_all_100.json'。
    """
    try:
        # 读取Excel文件
        df = pd.read_excel(excel_path, sheet_name=sheet_name)
        print(f"成功读取Excel文件: {excel_path}, Sheet: {sheet_name}")
    except FileNotFoundError:
        print(f"错误：未找到文件 {excel_path}。请检查文件路径是否正确。")
        return
    except Exception as e:
        print(f"读取Excel文件时发生错误: {e}")
        return

    conversations = []
    # 根据 max_entries 限制处理的行数
    rows_to_process = df.head(max_entries) if max_entries is not None else df

    for i, row in rows_to_process.iterrows():
        try:
            提问 = str(row['提问']) if pd.notna(row['提问']) else ""
            图片url = str(row['图片url']) if pd.notna(row['图片url']) else ""
            回答 = str(row['回答']) if pd.notna(row['回答']) else ""

            # 构建用户对话内容
            user_value = f"评价这张图片: {提问}<|vision_start|>{图片url}<|vision_end|>"

            conversation_entry = {
                "id": f"identity_{i+1}",
                "conversations": [
                    {
                        "from": "user",
                        "value": user_value
                    },
                    {
                        "from": "assistant",
                        "value": 回答
                    }
                ]
            }
            conversations.append(conversation_entry)
        except KeyError as e:
            print(f"错误：Excel中缺少列 {e}。请确保包含 '提问', '图片url', '回答' 列。")
            return
        except Exception as e:
            print(f"处理第 {i+1} 行时发生错误: {e}")
            continue # 继续处理下一行

    # 保存为JSON文件
    try:
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
        print(f"成功生成JSON文件: {output_json_path}，共处理 {len(conversations)} 条数据。")
    except Exception as e:
        print(f"保存JSON文件时发生错误: {e}")

# --- 如何使用这个函数 ---
if __name__ == "__main__":
    excel_file = r'D:\work\tain\gemini数据.xlsx' # 注意：Windows路径建议使用原始字符串 (r'') 或双反斜杠 (\\)
    # excel_file = '/path/to/your/gemini数据.xlsx' # Linux/macOS 路径示例

    # 示例1：处理所有数据
    # process_excel_to_json(excel_file, output_json_path='data_self_all_full.json')

    # 示例2：限制处理前 500 条数据
    process_excel_to_json(excel_file, max_entries=500, output_json_path='data_self_all_100.json')

    # 示例3：如果你的Excel文件不是Sheet1，可以指定sheet_name
    # process_excel_to_json(excel_file, sheet_name='我的数据表', max_entries=100, output_json_path='my_custom_data.json')