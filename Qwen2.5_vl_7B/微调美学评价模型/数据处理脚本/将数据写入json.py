import os
import pandas as pd
import json

# --- 配置项 ---
# 请务必修改为你自己的文件路径
SOURCE_EXCEL_PATH = r"D:\work\tain\美学评价\data\xlsx\测试.xlsx"
# 输出结果JSON训练数据文件路径
OUTPUT_JSON_PATH = r'D:\work\tain\美学评价\data\xlsx\ceshigpu_train_data.json'

# 需要从表格中读取的列名并按照先后顺序和固定格式拼接起来
COLUMNS_TO_READ = [
    "图片路径", "构图设计得分", "构图设计", "视觉元素得分", "视觉元素","技术执行得分",
    "技术执行",  "想象创意得分","想象创意",  "主题传达得分","主题传达",
      "情感反应得分", "情感反应", "整体完形得分","整体完形",
      "综合得分","综合评价"
]

# 固定的用户对话文本
USER_QUESTION = """你是一位资深的数字艺术评论家和图像分析专家。你的任务是对我提供的任意一张图片进行一次全面、客观、结构化的专业评价。

# 核心指令
请严格遵循以下定义的8个维度、评分规则和输出格式，完成对图片的分析。

---

### **第一部分：评价维度定义**

你必须且只能使用以下8个维度进行评价。每个维度的分析应基于以下准则：

1.  **构图设计 (Composition Design):**
    * 评估画面中构图的平衡性、对比性、布局美感与节奏感。重点关注焦点的动态设置、设计中的统一性与和谐感。

2.  **视觉元素 (Visual Elements):**
    * 分析色彩、几何结构、空间组织与光照等元素的相互作用，判断其是否能优化视觉对比与结构清晰度。

3.  **技术执行 (Technical Execution):**
    * 考察媒介与材料的运用能力，包括笔触、对焦、曝光、光线处理，以及图像清晰度和分辨率等专业技术水准。

4.  **想象创意 (Imagination & Creativity):**
    * 分析作品在概念和执行上的独特性，关注其是否超越常规风格、体现想象力及创新突破。

5.  **主题传达 (Theme Conveyance):**
    * 评估主题的清晰度和表达效果，考量其是否有效传达叙事内容、文化意义或社会语境。

6.  **情感反应 (Emotional Response):**
    * 判断作品能否引发观者情感共鸣、吸引注意力，以及是否留下深刻的个人化印象。

7.  **整体完形 (Overall Gestalt/Cohesion):**
    * 从整体上评估图像的视觉吸引力与艺术影响力，看其是否通过元素的整合形成有意义且引人入胜的印象。

8.  **综合评价 (Overall Evaluation):**
    * **此维度为独立评分项。** 对图像整体美学效果的全面评价，结合视觉冲击力、主题传达与艺术深度等多个方面进行综合判断，并给出最终分数。

---

### **第二部分：评分规则**

1.  **评分标准:** 10分制。
    * `9.00 - 10.00`: 杰作级别，几乎在所有方面都表现完美。
    * `7.50 - 8.99`: 非常优秀，技术和创意突出，有强烈的吸引力。
    * `6.00 - 7.49`: 良好，具备专业水准，但存在一些可以改进的方面。
    * `4.00 - 5.99`: 中等水平，基本概念和执行尚可，但有明显缺陷。
    * `0.00 - 3.99`: 较差，在多个核心维度上存在严重问题。
2.  **分数精度:** 所有8个维度的分数都必须精确到小数点后两位（例如：`8.39`）。
3.  **理由要求:** 每个维度的“理由”都应简洁、客观、专业，并直接引用画面中的具体元素来支撑你的评分。

---

### **第三部分：强制输出格式**

你的最终回答必须严格按照以下格式组织，不能有任何偏差，无需任何额外的问候语或解释。

构图设计得分: [此处填写分数]
构图设计：[此处填写具体理由]

视觉元素得分: [此处填写分数]
视觉元素：[此处填写具体理由]

技术执行得分: [此处填写分数]
技术执行：[此处填写具体理由]

想象创意得分: [此处填写分数]
想象创意：[此处填写具体理由]

主题传达得分: [此处填写分数]
主题传达：[此处填写具体理由]

情感反应得分: [此处填写分数]
情感反应：[此处填写具体理由]

整体完形得分: [此处填写分数]
整体完形：[此处填写具体理由]

综合评价得分: [此处填写分数]
综合评价：[此处填写综合得分理由]

"""

# --- 函数定义 ---

def construct_prompt(row):
    """
    根据DataFrame的行数据，构造完整的提示词，并按照指定格式在每个维度之间空一行。
    参数:
        row (pd.Series): DataFrame中的一行数据。
    返回:
        str: 拼接好的完整提示词。
    """
    prompt_lines = []
    
    # 遍历列表，步长为2，处理成对的得分和理由
    # 我们从索引1开始，跳过"图片名"
    for i in range(1, len(COLUMNS_TO_READ), 2):
        score_col = COLUMNS_TO_READ[i]
        reason_col = COLUMNS_TO_READ[i+1]

        score = row.get(score_col, "缺失数据")
        reason = row.get(reason_col, "缺失数据")

        # 拼接得分和理由
        prompt_lines.append(f"{score_col}: {score}")
        prompt_lines.append(f"{reason_col}: {reason}")
        
        # 在每对之后添加一个空行
        prompt_lines.append("")
    
    # 拼接所有行
    full_prompt = "\n".join(prompt_lines)
    return full_prompt


# --- 主程序 ---

def main():
    """
    主函数，负责读取Excel文件，处理数据，并生成JSON文件。
    """
    try:
        # 检查文件是否存在
        if not os.path.exists(SOURCE_EXCEL_PATH):
            print(f"错误：文件路径不存在 -> {SOURCE_EXCEL_PATH}")
            return

        # 使用pandas读取Excel文件，header=0表示第一行是列名
        df = pd.read_excel(SOURCE_EXCEL_PATH, header=0)
        print(f"成功读取Excel文件: {SOURCE_EXCEL_PATH}")

        conversations = []
        # 遍历DataFrame的每一行
        for i, row in df.iterrows():
            try:
                # 获取图片名，用作图片路径。如果为空则跳过。
                image_name = row.get("图片路径")
                if pd.isna(image_name):
                    print(f"警告：第 {i+1} 行缺少'图片路径'，已跳过。")
                    continue
                
                # 构建用户对话内容
                user_value = f"评价这张图片: {USER_QUESTION}<|vision_start|>{image_name}<|vision_end|>"
                
                # 调用函数生成助手响应文本
                assistant_value = construct_prompt(row)

                # 构建JSON条目
                conversation_entry = {
                    "id": f"identity_{i+1}",
                    "conversations": [
                        {
                            "from": "user",
                            "value": user_value
                        },
                        {
                            "from": "assistant",
                            "value": assistant_value
                        }
                    ]
                }
                conversations.append(conversation_entry)

            except Exception as e:
                print(f"处理第 {i+1} 行时发生错误: {e}")
                continue # 继续处理下一行

        # 保存为JSON文件
        with open(OUTPUT_JSON_PATH, 'w', encoding='utf-8') as f:
            json.dump(conversations, f, ensure_ascii=False, indent=2)
        print(f"成功生成JSON文件: {OUTPUT_JSON_PATH}，共处理 {len(conversations)} 条数据。")

    except Exception as e:
        print(f"处理文件时发生错误: {e}")


# 确保脚本被直接执行时运行main函数
if __name__ == "__main__":
    main()