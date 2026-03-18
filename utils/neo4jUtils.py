"""
    neo4j相关的工具函数
"""

import asyncio
from neo4j import AsyncGraphDatabase

class Neo4jUtils:
    def __init__(self, uri: str = "bolt://localhost:7687", auth: tuple = ("neo4j", "12345678")):
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

    
    def json_to_node_relationships(self, json_data):
        """
        将JSON数据转换为Neo4j节点和关系的格式

        :param json_data: 输入的JSON数据，包含节点和关系信息
        :return: 包含节点和关系的列表
        """
        
        nodes  = []
        triples = []

        # 处理实体，创建节点
        for entity, etype in json_data["entity_types"].items():
            nodes.append({"name": entity, "type": etype ,"timeStamp": json_data["entity_timeStamps"].get(entity, "unknown"),"level": 2})

        # 处理属性，创建节点和关系
        for subj, attrVals in json_data["attributes"].items():
            for attrVal in attrVals:
                nodes.append({"name": attrVal, "type": "attribute", "timeStamp": json_data["entity_timeStamps"].get(attrVal, "unknown"), "level": 1})
                triples.append({"source": subj, "target": attrVal, "relation": "HAS_ATTRIBUTE"})

        # 处理三元组，创建关系
        for tri in json_data["triples"]:
            sname, relRaw, oname = tri
            triples.append({"source": sname, "target": oname, "relation": relRaw})

        return nodes, triples