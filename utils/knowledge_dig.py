import json
import logging

from langchain_core.documents import Document

from .prompts import PROMPT_TEMPLATES
from .call_LLM_API import call_LLM_API
from .neo4jUtils import Neo4jUtils
from .milvusUtils import MilvusUtils

async def knowledge_storage(
        llm_api: call_LLM_API,
        llm_response: str,
        llm_result: dict,
        neo4j_utils: Neo4jUtils,
        nodeMilvus: MilvusUtils,
        tripleMilvus: MilvusUtils,
        chunkMilvus: MilvusUtils,
        question,
        options,
        schema
    ):
    knowledge_graph_prompt = PROMPT_TEMPLATES["knowledge_graph_extraction"].format(
        chunk = f"question: {question}\noptions: {options}\nLLM_response: {llm_response}",
        schema = json.dumps(schema)
    )
    kg_response = await llm_api.call_async(knowledge_graph_prompt)
    try:
        kg_json = json.loads(kg_response)
    except json.JSONDecodeError:
        logging.error("知识图谱提取失败, LLM返回的结果无法解析为JSON: %s", kg_response)
        return 0

    nodes,triples = neo4j_utils.json_to_node_relationships(kg_json)

    # 存储节点和三元组
    nodeMilvus.add_documents([Document(page_content=node["name"]) for node in nodes])

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
        if key == "questionBackground":
            documents.append(Document(page_content=f"{value}"))
        if key == "reasoning":
            documents.append(Document(page_content=f"{value}"))    
    # 存储chunk级别的上下文信息
    chunkMilvus.add_documents(documents)