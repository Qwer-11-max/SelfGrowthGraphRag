"""
    封装milvus, 后续可以添加一些检索算法
"""

from langchain_milvus import Milvus


class MilvusUtils:
    def __init__(   self, 
                    embedding_model:object,
                    host:str = "localhost",
                    port:str = "19530",
        ) -> None:
        
        self.embedding_model = embedding_model
        self.collection_name = "my_collection"
        self.connection_args = {
            "host": host,
            "port": port,
        }
        # 配置索引参数
        self.index_params = {
            "metric_type": "IP",        # 相似度度量：IP(内积，对应Cosine)、L2(欧式距离)
            "index_type": "IVF_FLAT",   # 索引类型：IVF_FLAT、HNSW、FLAT
            "params": {
                "nlist": 128           # IVF 参数：聚类中心数；HNSW 用 {"M": 16, "efConstruction": 200}
            }
        }


    # 在异步上下文中初始化 Milvus，并启用异步客户端。
    def get_async_milvus(   self,
                            collection_name: str,
                            drop_old: bool = False,
                            collection_description: str = "",
                            auto_id: bool = True,
                            consistency_level: str = "Strong") -> Milvus:
        """
        在异步上下文中初始化 Milvus，并启用异步客户端。
        注意：use_async 作为 Milvus 顶层参数传入，不要放到 connection_args 里。
        """
        vector_store = Milvus(
            embedding_function=self.embedding_model,
            collection_name=collection_name,
            connection_args=self.connection_args,
            index_params=self.index_params,
            collection_description=collection_description,
            drop_old=drop_old,
            auto_id=auto_id,
            consistency_level=consistency_level,
        )
        return vector_store