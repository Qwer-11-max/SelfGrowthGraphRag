import os
import asyncio
from openai import AsyncOpenAI
from tqdm.asyncio import tqdm_asyncio  # 异步专用的进度条
from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

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

# 初始化嵌入模型
embeddings = OllamaEmbeddings(
    model="nomic-embed-text:v1.5",  # 嵌入模型名称
    base_url="http://localhost:11434",  # Ollama服务地址
    temperature=0,
    num_ctx=4096,
)

# 在异步上下文中初始化 Milvus，并启用异步客户端。
def init_milvus_vector_store_async() -> Milvus:
    """
    在异步上下文中初始化 Milvus，并启用异步客户端。
    注意：use_async 作为 Milvus 顶层参数传入，不要放到 connection_args 里。
    """
    vector_store = Milvus(
        embedding_function=embeddings,
        collection_name="my_collection",
        connection_args={
            "host": "localhost",
            "port": "19530",
        },
        collection_description="知识库",
        drop_old=True,
        auto_id=True,
        consistency_level="Strong",
    )
    return vector_store

async def async_call_openai(index: int, prompt: str) -> tuple[int, str]:
    """
    异步调用 OpenAI API
    返回 (index, 结果)，方便进度条和结果对应
    """
    try:
        response = await async_client.chat.completions.create(
            model="qwen3.5-plus-2026-02-15",
            extra_body={
                "enable_thinking": False
            },
            response_format={"type": "json_object"},
            messages=[{"role": "user", "content": prompt}],
            timeout=60  # 设置超时时间，避免无限等待
        )
        return (index, response.choices[0].message.content)
    except Exception as e:
        return (index, f"调用失败: {str(e)}")

async def async_call_openai_limited(
        index: int, 
        question: str,
        options: str,
        answerMap: dict,
        semaphore: asyncio.Semaphore, 
        milvusDB: Milvus
    ) -> tuple[int, str]:
    """
    使用信号量限制并发请求数量。
    """
    async with semaphore:

        emb = embeddings.embed_query(f"{question} {options}")  # 生成查询向量
        content = await milvusDB.asimilarity_search_with_score_by_vector(embedding=emb,k=10)
        
        promptDemo = """
        你是一个专业的问答系统，以下是一个问题和选项，请你分析并给出正确答案：

        问题: {question}

        选项: {options}

        上下文:{context}

        请你根据问题,选项以及上下文，分析并给出正确答案，若是上下文无法提供足够判断的依据，
        那么你可以结合自身的知识做出回答。

        答案应该是选项中的一个。

        请按如下格式返回JSON结果:
        {
            "answer": "选项文本",
            "questionBackground": "对问题背景的理解与分析",
            "entityInfo":
                {
                    "entity1": "问题与选项中关键实体1的相关核心信息",
                    "entity2": "问题与选项中关键实体2的相关核心信息",
                    "...": "..."
                },
            "reasoning": "你的简要推理过程和分析，说明你是如何得出答案的"
        }
        """

        prompt = promptDemo.replace("{question}", question).replace("{options}", options)
        
        if len(content) == 0:
            pass  # 如果没有相关内容，直接调用模型回答问题
        else:
            contexts = []
            for item in content:
                doc = item[0]  # 文档对象
                score = item[1]  # 相似度分数
                context = f"\n\n相关内容（相似度: {score:.4f}）:\n{doc.page_content}"
                contexts.append(context)

            prompt = prompt.replace("{context}", "\n".join(contexts))
        
        index,json_string = await async_call_openai(index, prompt)
        try:
            resp = json.loads(json_string)
        except json.JSONDecodeError:
            return 0, json_string  # 如果解析失败，返回0和原始字符串以供调试
        
        if answerMap.get(resp.get("answer", ""), False):
            documents = []
            for key, value in resp.items():
                if key == "entityInfo":
                    entity_info_str = [f"{k}: {v}" for k, v in value.items()]
                    for doc in entity_info_str:
                        documents.append(Document(page_content=f"{doc}", metadata={"source": f"question_{index}_entity_info"}))
                if key == "questionBackgrand":
                    documents.append(Document(page_content=f"{value}", metadata={"source": f"question_{index}_background"}))
                if key == "reasoning":
                    documents.append(Document(page_content=f"{value}", metadata={"source": f"question_{index}_reasoning"}))
            
            milvusDB.add_documents(documents)
            return (1,json_string)  # 正确答案返回1，并返回prompt以供调试
        else:
            return (0,json_string)  # 错误答案返回0，并返回prompt以供调试

async def main():
    csv_path = r"/mnt/e/repository/data/datasets/ibm-research/FailureSensorIQ/single_true_multi_choice_qa.csv"
    all_results = extract_all_questions_data(csv_path)

    milvusDB = init_milvus_vector_store_async()
    print("Milvus 初始化成功:", milvusDB.collection_name)
    
    # all_results = all_results[:1000]  # 只处理前50条数据，方便测试和展示进度条效果

    questions = [res['question'] for res in all_results]
    options_list = [ "[" + ", ".join(res["options"]) + "]" for res in all_results]
    ansewerMap_list = [res['answer_mapping'] for res in all_results]

    # 控制同一时刻的最大并发请求数,默认为10，可以通过环境变量 MAX_CONCURRENCY 调整

    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)

    # 创建异步任务列表
    tasks = [async_call_openai_limited(i, q, o, a, semaphore, milvusDB) for i, (q, o, a) in enumerate(zip(questions, options_list, ansewerMap_list))]
    
    # 带进度条执行所有异步任务
    # desc: 进度条描述文字，colour: 进度条颜色（支持 red/green/blue/yellow 等）
    results = await tqdm_asyncio.gather(
        *tasks,
        desc="正在调用 Dashscope API",
        colour="green"
    )

    correct_count = 0
    for i,json_string in results:
        correct_count += i
    accuracy = correct_count / len(results) if results else 0
    print(f"\n===== 正确率 =====\nAccuracy: {accuracy:.2%}")

if __name__ == "__main__":
    asyncio.run(main())
