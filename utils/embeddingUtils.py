"""
    封装一些嵌入模型
"""

from langchain_ollama import OllamaEmbeddings

class embeddingUtils:
    def __init__(self):
        self.embeddings = OllamaEmbeddings(
            model="nomic-embed-text:v1.5",  # 嵌入模型名称
            base_url="http://localhost:11434",  # Ollama服务地址
            temperature=0,
            num_ctx=4096,
        )

    def embed_text(self, text,dim=768):
        """
        获取文本的嵌入向量

        :param text: 输入文本
        :param dim: 嵌入维度,默认为768
        :return: 嵌入向量, numpy数组, shape (dim,)
        """
        embedding = self.embeddings.embed_query(text)
        return embedding[:dim]  # 确保返回的嵌入向量维度为768
    
    def get_embeding_model(self):
        """
        获取当前使用的嵌入模型名称

        :return: 嵌入模型本身
        """
        return self.embeddings