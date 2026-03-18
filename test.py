# with open("./documents/knowledgeDemo/knowledge_base_100_en.txt", "r") as f:
#     text = f.read()

# import json
# with open("./schemas/knowledgeDemo/schema.json", "r") as f:
#     schema = json.load(f)

# from langchain_text_splitters import RecursiveCharacterTextSplitter, SpacyTextSplitter, TokenTextSplitter

# try:
#     text_splitter = TokenTextSplitter(chunk_size=4096, chunk_overlap=512)
# except ImportError:
#     text_splitter = RecursiveCharacterTextSplitter(chunk_size=4096, chunk_overlap=512)
# texts = text_splitter.create_documents(texts=[text])
# # print(texts[0])
# chunk = [text.page_content for text in texts][0]
# prompt = f"""
# 你是专家信息提取器和结构化数据组织者。
# 你的任务是分析提供的文本，尽可能多地提取有价值的实体、属性和关系。

# 指南：
# 1. 优先使用以下预定义模式：{schema}
# 2. 灵活性：如果上下文不符合预定义模式，则根据需要提取有价值的知识
# 3. 简洁性：属性和三元组应互补，无语义冗余
# 4. 不要错过任何有用信息
# 5. 模式演化:如果发现重要的新类型,添加到new_schema_types
# 6. 时间戳:为每个实体、属性和关系分配一个时间戳,表示和他们有关的时间,格式为"YYYY.MM.DD",
#     如果没有明确的时间,可以根据上下文推断一个合理的时间,或者是直接置为"unknown"
# 7. 提取时不用翻译,保持原文,但在返回的JSON中,属性和关系的命名要尽量规范化,去掉冗余词汇,保留核心语义

# 文本：{chunk}

# 有如下样例:
# {{
#     "attributes": {{"电压传感器": ["生产于2005年", "型号为X100"]}},
#     "triples": [["电压传感器", "安装在", "变电站A"]],
#     "entity_types": {
#         {"电压传感器": "device", "变电站A": "location"}
#     },
#     "entity_timeStamps": {
#         {"电压传感器": "2005.1.1"},
#         {"生产于2005年": "2005.1.1"},
#         {"安装在": "2005.1.1"}
#     },
#     "new_schema_types": {{"nodes": ["device"], "relations": ["安装在"], "attributes": ["生产于", "型号为"]}}
# }}

# 按照样例返回JSON格式:
# {{
#   "attributes": {{"实体": ["属性"]}},
#   "triples": [["实体1", "关系", "实体2"]],
#   "entity_types": {{"实体": "类型"}},
#   "entity_timeStamps": {
#         {"实体": "时间戳"},
#         {"属性": "时间戳"},
#         {"关系": "时间戳"}
#     },
#   "new_schema_types": {{"nodes": [], "relations": [], "attributes": []}}
# }}
# """

# import os
# from openai import OpenAI

# client = OpenAI(
#     api_key=os.getenv("DASHSCOPE_API_KEY"),
#     base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
# )

# resp = client.chat.completions.create(
#         model="qwen3.5-plus-2026-02-15",
#         extra_body={
#             "enable_thinking": False,
#             # "enable_search": True
#         },
#         response_format={"type": "json_object"},
#         messages=[{"role": "user", "content": prompt}],
#         timeout=60  # 设置超时时间，避免无限等待
# )

# results = resp.choices[0].message.content
# print(results)

import json
with open("test.json", "r") as f:
    test = json.load(f)

nodes  = []
triples = []

for entity, etype in test["entity_types"].items():
    nodes.append({"name": entity, "type": etype ,"timeStamp": test["entity_timeStamps"].get(entity, "unknown"),"level": 2})

for subj, attrVals in test["attributes"].items():
    for attrVal in attrVals:
        nodes.append({"name": attrVal, "type": "attribute", "timeStamp": test["entity_timeStamps"].get(attrVal, "unknown"), "level": 1})
        triples.append({"source": subj, "target": attrVal, "relation": "HAS_ATTRIBUTE"})

for tri in test["triples"]:
    sname, relRaw, oname = tri
    triples.append({"source": sname, "target": oname, "relation": relRaw})

node_insert_query = """
// 批量插入节点（去重）
CALL apoc.periodic.iterate(
  "UNWIND $payload.nodes AS n RETURN n",
  "
  WITH n,
       CASE 
         WHEN n.type IS NULL THEN ['UnknownLabel'] 
         ELSE [n.type] 
       END AS labels
  CALL apoc.merge.node(
    labels,
    {name: n.name},
    {timeStamp: coalesce(n.timeStamp, 'unknown'), level: coalesce(n.level, 2)},
    {}
  ) YIELD node
  RETURN count(node)
  ",
  {batchSize: 1000, parallel: true, params: {payload: $payload}}
);
"""

relationship_insert_query = """
// 批量插入关系（同时兜底创建缺失节点）
CALL apoc.periodic.iterate(
  "UNWIND $payload.rels AS r RETURN r",
  "
  MERGE (s:Entity {name: r.source})
  ON CREATE SET s.type = 'unknown', s.timeStamp = 'unknown', s.level = 2

  MERGE (t:Entity {name: r.target})
  ON CREATE SET
    t.type = CASE WHEN r.relation = 'HAS_ATTRIBUTE' THEN 'attribute' ELSE 'unknown' END,
    t.timeStamp = coalesce(r.timeStamp, 'unknown'),
    t.level = CASE WHEN r.relation = 'HAS_ATTRIBUTE' THEN 1 ELSE 2 END

  CALL apoc.merge.relationship(
    s,
    CASE
      WHEN r.relation IS NULL OR trim(r.relation) = '' THEN 'RELATED'
      ELSE trim(r.relation)
    END,
    {},  // identProps
    {timeStamp: coalesce(r.timeStamp, 'unknown')}, // onCreate
    t,
    {timeStamp: coalesce(r.timeStamp, 'unknown')}  // onMatch
  ) YIELD rel
  RETURN count(rel)
  ",
  {batchSize: 1000, parallel: false, params: {payload: $payload}}
);
"""

import asyncio
from neo4j import AsyncGraphDatabase

class AsyncNeo4jConnection:
    """异步 Neo4j 连接"""
    
    def __init__(self, uri: str, auth: tuple):
        self.driver = AsyncGraphDatabase.driver(uri, auth=auth)
    
    async def close(self):
        await self.driver.close()
    
    async def execute_query(self, query: str, parameters: dict = None):
        async with self.driver.session() as session:
            result = await session.run(query, parameters or {})
            return [record.data() async for record in result]
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

from langchain_ollama import OllamaEmbeddings

import faiss
import numpy as np

def compute_pairwise_cosine_similarity_faiss(vectors):
    """
    使用FAISS计算向量间两两余弦相似度
    
    参数:
        vectors: numpy数组, shape (n_vectors, dim)
    
    返回:
        similarity_matrix: 对称相似度矩阵, shape (n_vectors, n_vectors)
    """
    n_vectors, dim = vectors.shape
    
    # 1. 向量标准化（L2归一化）
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    normalized_vectors = vectors / norms
    
    # 2. 创建FAISS索引（使用内积）
    index = faiss.IndexFlatIP(dim)  # IP = Inner Product
    
    # 3. 添加向量到索引
    index.add(normalized_vectors.astype('float32'))
    
    # 4. 搜索自身（k=n_vectors 获取所有向量的相似度）
    k = n_vectors
    similarities, indices = index.search(normalized_vectors.astype('float32'), k)
    
    # 5. 构建相似度矩阵
    similarity_matrix = np.zeros((n_vectors, n_vectors))
    for i in range(n_vectors):
        # 按原始顺序重新排列
        sorted_order = np.argsort(indices[i])
        similarity_matrix[i] = similarities[i][sorted_order]
    
    return similarity_matrix

# 使用示例
async def main_async():
    async with AsyncNeo4jConnection("neo4j://localhost:7687", ("neo4j", "12345678")) as conn:
        # await conn.execute_query(node_insert_query, {"payload": {"nodes": nodes}})
        # await conn.execute_query(relationship_insert_query, {"payload": {"rels": triples}})

        data = await conn.execute_query("MATCH (n) RETURN n.name AS name")
        names = [record["name"] for record in data]
        embeddings = OllamaEmbeddings(
            model="nomic-embed-text-v2-moe",  # 嵌入模型名称
            base_url="http://localhost:11434",  # Ollama服务地址
            temperature=0,
            num_ctx=4096,
        )

        embeddings_list = []
        for name in names:
            embedding = embeddings.embed_query(name)
            embeddings_list.append(embedding)
        print(len(embeddings_list))
        vectors = np.array(embeddings_list).astype("float32")
        similarities = compute_pairwise_cosine_similarity_faiss(vectors)
        print(similarities)

        new_rels = []
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if similarities[i][j] > 0.95:  # 设置相似度阈值
                    new_rels.append({"source": names[i], "target": names[j], "relation": "EXTREAM_SIMILAR_TO"})
                elif similarities[i][j] > 0.85:
                    new_rels.append({"source": names[i], "target": names[j], "relation": "SIMILAR_TO"})
                elif similarities[i][j] > 0.80:
                    new_rels.append({"source": names[i], "target": names[j], "relation": "RELATED_TO"})

        print(f"发现 {len(new_rels)} 条新关系")
        await conn.execute_query(relationship_insert_query, {"payload": {"rels": new_rels}})
# 运行异步代码
asyncio.run(main_async())