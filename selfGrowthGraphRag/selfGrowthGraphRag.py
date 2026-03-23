import os
import sys

# 允许直接运行该脚本时找到项目根目录下的 utils 包。
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils.call_LLM_API import call_LLM_API
from utils.failure_IQ_read import failure_IQ_read
from utils.prompts import PROMPT_TEMPLATES
from utils.neo4jUtils import Neo4jUtils
from utils.embeddingUtils import embeddingUtils
from utils.milvusUtils import MilvusUtils
from utils.knowledge_dig import knowledge_storage

import asyncio
import json
import logging

from langchain_milvus import Milvus
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from tqdm.asyncio import tqdm_asyncio  # 异步专用的进度条


# 日志开关：
# - True: 打印每一次 LLM 请求结果
# - False: 不打印详细日志
ENABLE_VERBOSE_LLM_LOG = os.getenv("ENABLE_VERBOSE_LLM_LOG", "false").lower() in {"1", "true", "yes", "on"}


def log_llm_response(step: str, response: str) -> None:
    if ENABLE_VERBOSE_LLM_LOG:
        logging.info("[LLM][%s] response:\n%s", step, response)

async def process_question( 
        question,
        options,
        answer_mapping,
        semaphore: asyncio.Semaphore,
        schema: dict,
        llm_api: call_LLM_API,
        neo4j_utils: Neo4jUtils,
        embedding : OllamaEmbeddings,
        nodeMilvus: Milvus,
        tripleMilvus: Milvus,
        chunkMilvus: Milvus
    ):
    async with semaphore:
        # 0. 问题分解
        decomposed_prompt = PROMPT_TEMPLATES["question_decomposition_by_schema"].format(
            question=question, 
            schema=json.dumps(schema)
        )
        resp = await llm_api.call_async(decomposed_prompt)
        log_llm_response("question_decomposition_by_schema", resp)
        
        try:
            decomposition_result = json.loads(resp)
        except json.JSONDecodeError:
            logging.error("问题分解失败, LLM返回的结果无法解析为JSON: %s", resp)
            return 0
        
        # 1. 查询相关节点和三元组
        retrieved_nodes = []
        retrieved_triples = []
        retrieved_chunks = []

        for sub_q in decomposition_result.get("sub_questions", []):
            nodes = await nodeMilvus.asimilarity_search_with_score_by_vector(embedding=embedding.embed_query(sub_q), k=10)
            triples = await tripleMilvus.asimilarity_search_with_score_by_vector(embedding=embedding.embed_query(sub_q), k=10)
            chunks = await chunkMilvus.asimilarity_search_with_score_by_vector(embedding=embedding.embed_query(sub_q), k=3)
            retrieved_nodes.extend([node for node, _ in nodes])
            retrieved_chunks.extend([chunk for chunk, _ in chunks])
            retrieved_triples.extend([triple for triple, _ in triples])

        # 2. 构建询问LLM的prompt，并请求调用LLM接口
        request_prompt = PROMPT_TEMPLATES["question_with_context"].format(
            question= question,
            options = options,
            context = {
                "retrieved_nodes": [node.page_content for node in retrieved_nodes],
                "retrieved_triples": [triple.page_content for triple in retrieved_triples],
                "retrieved_chunks": [chunk.page_content for chunk in retrieved_chunks]
            } 
        )
        llm_response = await llm_api.call_async(request_prompt)
        log_llm_response("question_with_context", llm_response)

        # 3. 解析LLM返回的结果，将新的节点和关系存入Neo4j和Milvus中
        try:
            llm_result = json.loads(llm_response)
        except json.JSONDecodeError:
            logging.error("LLM返回的结果无法解析为JSON: %s", llm_response)
            return 0
        
        if answer_mapping.get(llm_result.get("answer", ""), False):
            # 动态知识挖掘
            await knowledge_storage(
                llm_api=llm_api,
                llm_response=llm_response,
                llm_result=llm_result,
                neo4j_utils=neo4j_utils,
                nodeMilvus=nodeMilvus,
                tripleMilvus=tripleMilvus,
                chunkMilvus=chunkMilvus,
                question=question,
                options=options,
                schema=schema
            )
            return 1  # 正确答案返回1，并返回prompt以供调试
        else:
            return 0  # 错误答案返回0，并返回prompt以供调试

async def main():
    logging.basicConfig(
        level=logging.INFO if ENABLE_VERBOSE_LLM_LOG else logging.WARNING,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )
    logging.info("ENABLE_VERBOSE_LLM_LOG=%s", ENABLE_VERBOSE_LLM_LOG)

    #============== 准备数据 ==============
    csv_path = r"/mnt/e/repository/data/datasets/ibm-research/FailureSensorIQ/single_true_multi_choice_qa.csv"
    questionPd =  failure_IQ_read(csv_path).get_data()
    print(f"数据集加载完成,共 {len(questionPd)} 条数据。")

    questionPd = questionPd[:1000] 

    questions = [res["question"] for res in questionPd]
    options_list = ["[" + ", ".join(res["options"]) + "]" for res in questionPd]
    ansewerMap_list = [res["answer_mapping"] for res in questionPd]

    with open("./schemas/knowledgeDemo/schema.json", "r") as f:
        schema = json.load(f)
    
    #============== 初始化工具类 ==============
    llm_api = call_LLM_API()
    neo4j_utils = Neo4jUtils()
    embedding_utils = embeddingUtils()
    milvus = MilvusUtils(embedding_model=embedding_utils.get_embeding_model())
    nodeMilvus = milvus.get_async_milvus(collection_name="nodes",drop_old=True)
    tripleMilvus = milvus.get_async_milvus(collection_name="triples",drop_old=True)
    chunkMilvus = milvus.get_async_milvus(collection_name="chunks",drop_old=True)

    #=============== 组装任务 ==============
    MAX_CONCURRENCY = 20  # 最大并发数，根据实际情况调整
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    tasks = []
    for question, options, answer_mapping in zip(questions, options_list, ansewerMap_list):
        task = process_question(
            question=question,
            options=options,
            answer_mapping=answer_mapping,
            semaphore=semaphore,
            schema=schema,
            llm_api=llm_api,
            neo4j_utils=neo4j_utils,
            embedding = embedding_utils.get_embeding_model(),
            nodeMilvus=nodeMilvus,
            tripleMilvus=tripleMilvus,
            chunkMilvus=chunkMilvus
        )
        tasks.append(task)

    res = await tqdm_asyncio.gather(*tasks)

    correct_count = sum(res)
    accuracy = correct_count / len(res) if res else 0
    print(f"\n===== 正确率 =====\nAccuracy: {accuracy:.2%}")

    #=============== 关闭Neo4j连接 ==============
    await neo4j_utils.close()

    #============== 打印token用量 ==============
    print("\n===== LLM Token Usage =====")
    llm_api.print_total_tokens_used()

#============== 处理每一行数据 ==============
if __name__ == "__main__":
    asyncio.run(main())