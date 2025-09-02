import os
import time
import pandas as pd
import logging

from DrissionPage import ChromiumPage, ChromiumOptions

# --- 日志配置 ---
# 配置日志记录，方便在运行时查看进度和调试错误
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 配置参数 ---
# !!! 请务必修改为你自己的图片文件夹路径 !!!
IMAGE_FOLDER = r"C:\Users\f60055380\Pictures\1"  # 示例路径，请修改
TARGET_URL = 'https://artimuse.intern-ai.org.cn/'
OUTPUT_EXCEL_FILE = 'image_evaluation_results.xlsx'


# --- 数据提取核心函数 ---
def upload_image_and_extract_data(tab, image_path):
    """
    上传一张图片，并从结果页面提取评分数据。
    此函数经过修正，可精确定位总分、描述、评分细则等信息。
    """
    try:
        # 1. 上传图片
        logging.info(f"正在上传图片: {os.path.basename(image_path)}")
        file_input = tab.ele('xpath://input[@type="file"]')
        file_input.input(image_path)

        # 2. 等待评分结果页面加载
        logging.info("图片上传成功，正在等待评分结果...")
        
        # 根据您提供的完整HTML，id="scoreArea" 是一个稳定的定位点
        tab.wait.ele_displayed('#scoreArea', timeout=45) 
        # 然后再等待scoreArea内部的span元素出现
        tab.wait.ele_displayed('#scoreNum span', timeout=10) # 再次确保具体的span元素可见
        time.sleep(3) # 增加短暂的固定等待，确保所有JS渲染完成并稳定

        # 3. 开始提取数据
        logging.info(f"开始提取图片 '{os.path.basename(image_path)}' 的评估数据。")
        result_data = {"Image Name": os.path.basename(image_path)}

        # 提取总分 (已修正，直接等待稳定的 scoreNum 元素)
        try:
            # 等待 id 为 scoreNum 的元素出现
            tab.wait.ele_displayed('#scoreNum', timeout=45)
            time.sleep(5) 

            # 直接查找 scoreNum 下的 span 元素
            overall_score_element = tab.ele('#scoreNum span')
            if overall_score_element and overall_score_element.is_displayed():
                overall_score = overall_score_element.text
                result_data["Overall Score"] = overall_score
                logging.info(f"提取到总分: {overall_score}")
            else:
                raise ValueError("未能找到总分span元素或其不可见。")
        except Exception as e:
            logging.warning(f"未能提取到总分: {e}")
            result_data["Overall Score"] = "N/A"

        # 提取图片描述
        try:
            # 等待描述元素出现，增加稳定性
            tab.wait.ele_displayed('css:._summary_12p4y_97', timeout=10)
            summary_text = tab.ele('css:._summary_12p4y_97').text
            result_data["Image Description"] = summary_text
            logging.info(f"提取到图片描述: {summary_text[:30]}...")
        except Exception as e:
            logging.warning(f"未能提取到图片描述: {e}")
            result_data["Image Description"] = "N/A"

        # 截取维度图
        try:
            # 确保canvas元素已加载并可见
            tab.wait.ele_displayed('tag:canvas', timeout=10)
            canvas_element = tab.ele('tag:canvas')
            screenshot_name = os.path.splitext(os.path.basename(image_path))[0] + "_dimension_chart.png"
            screenshot_path = os.path.join(IMAGE_FOLDER, screenshot_name)
            canvas_element.get_screenshot(path=screenshot_path)
            result_data["Dimension Chart Path"] = screenshot_path
            logging.info(f"维度图已保存: {screenshot_path}")
        except Exception as e:
            logging.warning(f"截取维度图失败: {e}")
            result_data["Dimension Chart Path"] = "N/A"

        # 提取评分细则
        try:
            # 等待至少一个评分细则项目出现
            tab.wait.ele_displayed('css:._scoreDetails-item_12p4y_142', timeout=10)
            detail_items = tab.eles('css:._scoreDetails-item_12p4y_142')
            if not detail_items:
                logging.warning("未找到评分细则项目。")
            else:
                logging.info(f"找到 {len(detail_items)} 条评分细则，正在逐条提取...")
                for item in detail_items:
                    title = item.ele('css:._title_12p4y_148').text.strip()
                    text = item.ele('css:._text_12p4y_168').text.strip()
                    result_data[title] = text
                logging.info("所有评分细则提取完毕。")
        except Exception as e:
            logging.warning(f"提取评分细则时发生错误: {e}")

        return result_data

    except Exception as e:
        logging.error(f"处理图片 {os.path.basename(image_path)} 时发生严重错误: {e}")
        return {
            "Image Name": os.path.basename(image_path),
            "Overall Score": "Error",
            "Image Description": f"Processing error: {e}",
            "Dimension Chart Path": "N/A"
        }


# --- 主程序执行逻辑 ---
if __name__ == '__main__':
    page = None

    try:
        co = ChromiumOptions()
        # co.headless() # 如果需要无头模式，取消此行注释
        page = ChromiumPage(co)
        tab = page.new_tab()
        
        logging.info(f"正在访问目标网址: {TARGET_URL}")
        tab.get(TARGET_URL)

        try:
            tab.wait.ele_displayed('xpath://input[@type="file"]', timeout=10)
            logging.info("目标网页加载成功，已准备好上传文件。")
        except Exception as e:
            logging.error(f"页面初始化失败或未找到文件上传框: {e}")
            exit()

        all_image_results = []
        image_files = [os.path.join(IMAGE_FOLDER, f) for f in os.listdir(IMAGE_FOLDER)
                       if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

        if not image_files:
            logging.warning(f"在文件夹 '{IMAGE_FOLDER}' 中未找到任何图片。请检查路径和文件类型。")
        else:
            logging.info(f"共发现 {len(image_files)} 张图片，开始处理...")
            for i, image_path in enumerate(image_files):
                logging.info(f"--- 开始处理第 {i+1}/{len(image_files)} 张图片: {os.path.basename(image_path)} ---")

                # 每次处理新图片前，重新加载页面
                if i > 0: # 除了第一张图片，后续图片都需要重新加载页面
                    logging.info(f"重新加载页面以处理下一张图片...")
                    tab.get(TARGET_URL)
                    tab.wait.ele_displayed('xpath://input[@type="file"]', timeout=10)
                    time.sleep(2) # 增加短暂等待确保页面完全准备好

                result = upload_image_and_extract_data(tab, image_path)
                all_image_results.append(result)
                
                logging.info(f"图片 '{os.path.basename(image_path)}' 处理完毕。")


        if all_image_results:
            df = pd.DataFrame(all_image_results)
            try:
                df.to_excel(OUTPUT_EXCEL_FILE, index=False, engine='openpyxl')
            except Exception as e:
                logging.error(f"保存Excel文件时出错: {e}")
            finally:
                logging.info(f"任务完成！所有图片评估结果已保存至 '{OUTPUT_EXCEL_FILE}'")
        else:
            logging.info("没有收集到任何数据，无需保存Excel文件。")

    except Exception as overall_e:
        logging.critical(f"主程序运行期间发生未知严重错误: {overall_e}")

    finally:
        # 修正程序退出逻辑 (已修正)
        # 检查page对象是否被成功创建，然后关闭
        if page:
            page.quit()
            logging.info("浏览器已关闭。")