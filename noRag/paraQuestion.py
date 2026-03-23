import os
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio  # 异步专用的进度条

import pandas as pd
import json
import re

MAX_CONCURRENCY = int(os.getenv("MAX_CONCURRENCY", "10"))

def extract_all_questions_data(csv_path):
    df = pd.read_csv(csv_path)
    results = []
    
    for _, row in df.iterrows():
        # 处理选项
        # options_str = row['options'].strip("[]").replace("'", "")
        # options = [opt.strip() for opt in options_str.split()]
        
        # 处理正确答案
        correct_str = row['correct'].strip("[]")
        correct_values = [val.strip() == 'True' for val in correct_str.split()]

        # 更精确地处理选项（处理带空格的选项名称）
        options_str = row['options'].strip("[]").replace("'", "")
        
        # 使用正则表达式正确分割选项
        # 匹配被单引号包围的字符串
        pattern = r"'([^']+)'"
        options = re.findall(pattern, row['options'])
        
        # 如果正则匹配失败，回退到原来的分割方法
        if not options:
            # 将连续的空格替换为单个空格，然后按空格分割
            options = [opt for opt in options_str.split() if opt]
        # 构建选项-答案映射字典
        option_answer_dict = dict(zip(options, correct_values))
        
        results.append({
            'id': row['id'],
            'question': row['question'],
            'options': options,
            'answer_mapping': option_answer_dict
        })
    
    return results

# 初始化异步客户端
async_client = AsyncOpenAI(
    api_key=os.getenv("DASHSCOPE_API_KEY"),
    base_url=os.getenv("DASHSCOPE_BASE_URL", "https://api.dashscope.com/v1")
    )

async def async_call_openai(index: int, prompt: str) -> tuple[int, str]:
    """
    异步调用 OpenAI API
    返回 (index, 结果)，方便进度条和结果对应
    """
    try:
        response = await async_client.chat.completions.create(
            model="qwen3.5-plus-2026-02-15",
            extra_body={"enable_thinking": True},
            messages=[{"role": "user", "content": prompt}],
            timeout=120  # 设置超时时间，避免无限等待
        )
        return (index, response.choices[0].message.content.strip())
    except Exception as e:
        return (index, f"调用失败: {str(e)}")

async def async_call_openai_limited(index: int, prompt: str, semaphore: asyncio.Semaphore) -> tuple[int, str]:
    """
    使用信号量限制并发请求数量。
    """
    async with semaphore:
        return await async_call_openai(index, prompt)

async def main():
    csv_path = r"/mnt/e/repository/data/datasets/ibm-research/FailureSensorIQ/single_true_multi_choice_qa.csv"
    all_results = extract_all_questions_data(csv_path)

    all_results = all_results[:1000]  # 只处理前20条数据，方便测试和展示进度条效果

    questions = [res['question'] for res in all_results]
    options_list = [ "[" + ", ".join(res["options"]) + "]" for res in all_results]
    
    # 示例：更多的 prompt 更能体现进度条效果
    prommpt = """
    你是一个专业的问答系统，以下是一个问题和选项，请你分析并给出正确答案：
    
    问题: {question}
    
    选项: {options}

    请你根据问题和选项，分析并给出正确答案。答案应该是选项中的一个。
    
    请直接返回正确的选项文本，不要返回选项的字母标识。
    """
    
    prompts = [prommpt.format(question=q, options=o) for q, o in zip(questions, options_list)]

    # 控制同一时刻的最大并发请求数,默认为10，可以通过环境变量 MAX_CONCURRENCY 调整
    # MAX_CONCURRENCY = 30
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    # 创建异步任务列表
    tasks = [async_call_openai_limited(i, p, semaphore) for i, p in enumerate(prompts)]
    
    # 带进度条执行所有异步任务
    # desc: 进度条描述文字，colour: 进度条颜色（支持 red/green/blue/yellow 等）
    results = await tqdm_asyncio.gather(
        *tasks,
        desc="正在调用 Dashscope API",
        colour="green"
    )

    # 计算正确率
    correct_count = 0
    for index, result in results:
        if result in all_results[index]['options']:
            if all_results[index]['answer_mapping'].get(result, False):
                correct_count += 1

    accuracy = correct_count / len(results) if results else 0
    print(f"\n===== 正确率 =====\nAccuracy: {accuracy:.2%}")

if __name__ == "__main__":
    asyncio.run(main())


# 正确率61.64%，无RAG，单纯的问答系统，模型是qwen3.5-plus-2026-02-15，最大并发数10，处理了全部2667条数据。