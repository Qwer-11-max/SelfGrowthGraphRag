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
ENABLE_VERBOSE_LLM_LOG = os.getenv("ENABLE_VERBOSE_LLM_LOG", "true").lower() in {"1", "true", "yes", "on"}


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
            raise ValueError(f"问题分解失败,LLM返回的结果无法解析为JSON: {resp}")
        
        # 1. 查询相关节点和三元组
        retrieved_nodes = []
        retrieved_triples = []
        retrieved_chunks = []

        for sub_q in decomposition_result.get("sub_questions", []):
            nodes = await nodeMilvus.asimilarity_search_with_score_by_vector(embedding=embedding.embed_query(sub_q), k=3)
            triples = await tripleMilvus.asimilarity_search_with_score_by_vector(embedding=embedding.embed_query(sub_q), k=3)
            chunks = await chunkMilvus.asimilarity_search_with_score_by_vector(embedding=embedding.embed_query(sub_q), k=3)
            retrieved_nodes.extend(nodes)
            retrieved_chunks.extend(chunks)
            retrieved_triples.extend(triples)
            logging.info(f"子问题: {sub_q}\n")

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
            raise ValueError(f"LLM返回的结果无法解析为JSON: {llm_response}")
        
        if answer_mapping.get(llm_result.get("answer", ""), False):
            # 动态知识挖掘
            knowledge_graph_prompt = PROMPT_TEMPLATES["knowledge_graph_extraction"].format(
                chunk = f"question: {question}\noptions: {options}\nLLM_response: {llm_response}",
                schema = json.dumps(schema)
            )
            kg_response = await llm_api.call_async(knowledge_graph_prompt)
            log_llm_response("knowledge_graph_extraction", kg_response)
            try:
                kg_json = json.loads(kg_response)
            except json.JSONDecodeError:
                raise ValueError(f"知识图谱提取失败,LLM返回的结果无法解析为JSON: {kg_response}")

            nodes,triples = neo4j_utils.json_to_node_relationships(kg_json)

            # 存储节点和三元组
            nodeMilvus.add_documents([Document(page_content=node) for node in nodes])

            triDocs = []
            for triple in triples:
                triple_str = f"{triple['source']} {triple['relation']} {triple['target']}"
                triDocs.append(Document(page_content=triple_str,metadata={"source": triple["source"], "relation": triple["relation"], "target": triple["target"]}))
            tripleMilvus.add_documents(triDocs)

            documents = []
            for key, value in llm_result.items():
                if key == "entityInfo":
                    entity_info_str = [f"{k}: {v}" for k, v in value.items()]
                    for doc in entity_info_str:
                        documents.append(Document(page_content=f"{doc}"))
                if key == "questionBackgrand":
                    documents.append(Document(page_content=f"{value}"))
                if key == "reasoning":
                    documents.append(Document(page_content=f"{value}"))    
            # 存储chunk级别的上下文信息
            chunkMilvus.add_documents(documents)
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

    questionPd = questionPd[:2] # 取前10行数据进行测试，正式运行时可以注释掉这一行

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
    MAX_CONCURRENCY = 5  # 最大并发数，根据实际情况调整
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

#============== 处理每一行数据 ==============
if __name__ == "__main__":
    asyncio.run(main())