# -*- coding: utf-8 -*-

import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

# --- 核心依赖 ---
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from faiss import IndexFlatL2

# --- LangChain 组件  ---
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- 文档解析库 (通用) ---
try:
    import fitz  # PyMuPDF
    from docx import Document
    from pptx import Presentation
except ImportError as e:
    print(f"警告: 缺少文档解析库。详细信息: {e}")
    fitz, Document, Presentation = None, None, None



# 获取可用GPU数量
def get_num_gpus_from_env():
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if cuda_visible:
        return len([x for x in cuda_visible.split(',') if x.strip().isdigit()])
    try:
        import torch
        return torch.cuda.device_count()
    except Exception:
        return 1


class AdvancedRAGSummarizer:
    """
    一个集成了RAG、多级MapReduce和模型特定功能（如Qwen3的“思考模式”）的高级文档处理类。
    """

    def __init__(self, model_name_or_path: str, embedding_model_name: str = "Qwen/Qwen3-Embedding-8B", max_model_len: int = 40960, enable_thinking: bool = True):
        """
        初始化模型、分词器和RAG组件。
        """
        print("="*50)
        print("正在加载LLM模型和分词器...")
        try:
            num_gpus = get_num_gpus_from_env()
            print(f"检测到可用GPU数量: {num_gpus}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "{{ '<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n' }}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<|im_start|>assistant\n' }}"
                "{% endif %}"
            )
            self.llm = LLM(
                model=model_name_or_path,
                max_model_len=max_model_len,
                trust_remote_code=True,
                tensor_parallel_size=num_gpus if num_gpus > 1 else 1
            )
            self.enable_thinking = enable_thinking
            self.max_model_len = max_model_len
            print(f"LLM模型和分词器加载完成。Qwen3思考模式已{'启用' if self.enable_thinking else '禁用'}")
            print(f"正在加载Embedding模型: {embedding_model_name}...")
            self.embedding_llm = LLM(
                model=embedding_model_name,
                task="embed",
                tensor_parallel_size=num_gpus if num_gpus > 1 else 1
            )
            print("Embedding模型加载完成。")
            print("="*50)
        except Exception as e:
            raise RuntimeError(f"错误：加载模型失败。请检查路径和环境配置。详细信息: {e}")

    def _clean_response(self, text: str) -> str:
        """清理模型输出中的特殊标记。"""
        if text:
            # 清理 Qwen3 的思考模式标签
            text = re.sub(r'<\|im_end\|>', '', text, flags=re.DOTALL)
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            text = text.strip()
        return text

    def _get_sampling_params(self, is_thinking: bool):
        """根据是否启用思考模式返回不同的采样参数。"""
        if is_thinking:
            # 官方推荐的思考模式参数
            return SamplingParams(temperature=0.6, top_p=0.95, top_k=20, max_tokens=2048)
        else:
            # 官方推荐的非思考模式参数
            return SamplingParams(temperature=0.7, top_p=0.8, top_k=20, max_tokens=2048)

    def _recursive_summarize(self, texts: List[str], level: int = 1) -> str:
        """
        递归地对文本块进行总结，直到汇总结果小于模型长度限制。
        支持并发LLM推理以加速大批量文本块处理。
        """
        print(f"\n--- 第 {level} 级分块总结 (共 {len(texts)} 块) ---")
        prompts = []
        soft_switch = "/think" if self.enable_thinking else "/no_think"
        for text_chunk in texts:
            messages = [
                {"role": "user", "content": f"{soft_switch} 请根据以下内容片段，生成一个简洁但全面的小结，以便后续进行最终汇总。\n\n内容片段：\n{text_chunk}\n\n小结："}
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            prompts.append(prompt)

        sub_summary_params = self._get_sampling_params(self.enable_thinking)
        sub_summary_params.max_tokens = max(512, self.max_model_len // 10)

        # 支持并发LLM推理（如llm.generate本身不支持批量，可拆分并发）
        num_gpus = get_num_gpus_from_env()
        max_workers = min(32, len(prompts), num_gpus * 4 if num_gpus > 1 else 8)  # 多卡提升并发
        results = [None] * len(prompts)
        try:
            if hasattr(self.llm, 'generate') and getattr(self.llm.generate, '__self__', None) is self.llm and hasattr(self.llm, 'batch_generate'):
                # 如果llm支持batch_generate，优先用批量
                outputs = self.llm.batch_generate(prompts, sub_summary_params)
                sub_summaries = [self._clean_response(output.outputs[0].text) for output in outputs]
            else:
                # 否则用线程池并发
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    future_to_idx = {executor.submit(self.llm.generate, [prompt], sub_summary_params): i for i, prompt in enumerate(prompts)}
                    for future in as_completed(future_to_idx):
                        idx = future_to_idx[future]
                        try:
                            output = future.result()
                            results[idx] = self._clean_response(output[0].outputs[0].text)
                        except Exception as e:
                            results[idx] = f"[小结失败: {e}]"
                sub_summaries = results
            combined_summary = "\n\n---\n\n".join(sub_summaries)
            summary_tokens = self.tokenizer.encode(combined_summary)
            if len(summary_tokens) > self.max_model_len * 0.8:
                print(f"第{level}级汇总后Token数 {len(summary_tokens)} 超过限制，进行下一级汇总。")
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.max_model_len,
                    chunk_overlap=200,
                    length_function=lambda x: len(self.tokenizer.encode(x))
                )
                new_chunks = splitter.split_text(combined_summary)
                return self._recursive_summarize(new_chunks, level + 1)
            else:
                return combined_summary
        except Exception as e:
            raise RuntimeError(f"在第 {level} 级汇总时调用模型发生错误: {e}")

    # ... 其他方法保持不变 ...


    def process_document(self, file_path: str, question_text: str, chunk_size_chars: int = 1000, sheet_name: Optional[str] = None) -> str:
        if not os.path.exists(file_path):
            return f"错误：未找到文件 '{file_path}'。"

        try:
            print("\n--- 步骤1: 结构化分块表格内容 ---")
            # 判断是否为表格文件
            if file_path.endswith(('.xlsx', '.xls', '.csv')):
                if file_path.endswith('.csv'):
                    df = pd.read_csv(file_path)
                else:
                    df = pd.read_excel(file_path, sheet_name=sheet_name) if sheet_name else pd.read_excel(file_path)
                # 每行拼成结构化文本块
                row_blocks = []
                for idx, row in df.iterrows():
                    block = []
                    for col in df.columns:
                        block.append(f"{col}: {row[col]}")
                    row_blocks.append("\n".join(block))
                # 合并短块，保证每块 token 不太短
                text_chunks = []
                cur_chunk = ""
                for block in row_blocks:
                    if not cur_chunk:
                        cur_chunk = block
                    else:
                        # 合并后长度不超过 chunk_size_chars
                        if len(cur_chunk) + len(block) < chunk_size_chars:
                            cur_chunk += "\n---\n" + block
                        else:
                            text_chunks.append(cur_chunk)
                            cur_chunk = block
                if cur_chunk:
                    text_chunks.append(cur_chunk)
                print(f"表格已结构化分块为 {len(text_chunks)} 个块。")
            else:
                # 非表格文件，走原有文本提取与分块
                full_text = self._extract_text_from_file(file_path, sheet_name)
                splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size_chars, chunk_overlap=100, length_function=len)
                text_chunks = splitter.split_text(full_text)
                print(f"文本已分割成 {len(text_chunks)} 个语义块。")

            print("\n--- 步骤2: 构建向量索引并进行RAG检索 ---")
            start_time = time.time()

            print("正在生成文本块嵌入向量...")
            chunk_embed_outputs = self.embedding_llm.embed(text_chunks)
            chunk_vectors = [output.outputs.embedding for output in chunk_embed_outputs]

            question_instructed = f"Given a user's question, retrieve relevant passages that answer the question.\n{question_text}"
            question_embed_output = self.embedding_llm.embed([question_instructed])
            question_vector = question_embed_output[0].outputs.embedding

            d = len(chunk_vectors[0])
            chunk_vectors_np = np.array(chunk_vectors).astype('float32')

            index = IndexFlatL2(d)
            index.add(chunk_vectors_np)

            print(f"向量索引构建完成，用时: {time.time() - start_time:.2f} 秒")

            print(f"正在根据问题检索最相关的文本块...")
            D, I = index.search(np.array([question_vector]).astype('float32'), k=min(10, len(text_chunks)))
            retrieved_chunks = [text_chunks[i] for i in I[0]]
            print(f"成功检索到 {len(retrieved_chunks)} 个相关文本块。")

            all_sub_summaries_text = self._recursive_summarize(retrieved_chunks)

            print("\n--- 步骤3: 汇总所有小结，生成最终总结 ---")
            final_summary_instruction = f"我已从一个大型文档中检索出与问题相关的片段，并对它们进行了初步总结。以下是所有相关的小结内容，请根据这些内容，生成一个详细且连贯的最终总结。\n\n所有小结：\n{all_sub_summaries_text}"

            final_summary_instruction = f"{'/think' if self.enable_thinking else '/no_think'} " + final_summary_instruction
            final_prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": final_summary_instruction}], tokenize=False, add_generation_prompt=True)

            final_summary_params = self._get_sampling_params(self.enable_thinking)
            outputs = self.llm.generate([final_prompt], final_summary_params)
            final_summary = self._clean_response(outputs[0].outputs[0].text)

            print("\n--- 步骤4: 基于最终总结，回答用户最初的问题 ---")
            answer_instruction = f"现在，请根据以下这份总结报告，清晰、准确地回答我最初的问题。\n\n总结报告：\n{final_summary}\n\n我的问题是：'{question_text}'\n\n请回答："

            answer_instruction = f"{'/think' if self.enable_thinking else '/no_think'} " + answer_instruction
            answer_prompt = self.tokenizer.apply_chat_template([{"role": "user", "content": answer_instruction}], tokenize=False, add_generation_prompt=True)

            answer_params = self._get_sampling_params(self.enable_thinking)
            outputs = self.llm.generate([answer_prompt], answer_params)
            final_answer = self._clean_response(outputs[0].outputs[0].text)

            return f"✅ **处理完成！**\n\n" + "="*20 + " **综合总结** " + "="*20 + f"\n\n{final_summary}\n\n" + "="*20 + " **问题回答** " + "="*20 + f"\n\n{final_answer}"

        except (ValueError, ImportError, RuntimeError) as e:
            return f"处理过程中发生错误: {e}"

# --- 脚本主入口 ---
if __name__ == "__main__":
    llm_model_path = r"/DATA/f60055380/Qwen3_8B_TL/Model/qwen3-8b-hf"
    embedding_model = r"/DATA/f60055380/Qwen3_8B_TL/Model/Qwen3-Embedding-8B" 
    file_to_process = r"/DATA/f60055380/Qwen3_8B_TL/data/xlsx/问题集1200.xlsx"
    question = "总结不同维度不同分数的规律。"
    
    start_time = time.time()
    try:
        summarizer = AdvancedRAGSummarizer(
            model_name_or_path=llm_model_path,
            embedding_model_name=embedding_model,
            enable_thinking=True  # 设置为 True 或 False 来控制思考模式
        )
        final_output = summarizer.process_document(
            file_to_process,
            question,
            chunk_size_chars=1000,
            sheet_name="Sheet1" if file_to_process.endswith(('.xlsx', '.xls')) else None
        )
        
        print("\n\n" + "="*25 + " **最终输出结果** " + "="*25)
        print(final_output)
        print("="*70)
        
    except Exception as e:
        print(f"\n处理过程中发生致命错误: {e}")
    
    end_time = time.time()
    print(f"整个任务总用时: {end_time - start_time:.2f} 秒")