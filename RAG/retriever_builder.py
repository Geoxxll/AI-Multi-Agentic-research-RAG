import logging
from typing import List
from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_community.vectorstores.utils import maximal_marginal_relevance
import heapq
from langchain_openai import OpenAIEmbeddings
from typing import Dict
from langchain_cohere import CohereRerank

logger = logging.getLogger(__name__)
# BM25 -> samilarityEmbeddingSearch
class Retrievers:
    def __init__(self, chunked_doc: List[str], vectorstore: Chroma):
        self.chunked_doc = chunked_doc
        self.vectorstore = vectorstore
        self.cohere_rerank = CohereRerank(model="rerank-english-v3.0", top_n=4)

    def build_retriever(self):
        try:
            logger.info("Building BM25 retriever")
            bm25_retriever = BM25Retriever.from_documents(self.chunked_doc)
            bm25_retriever.k = 10

            logger.info("Building vector-based retrivers.")
            retriever_vanilla = self.vectorstore.as_retriever(
                search_type="similarity", search_kwargs={"k": 10}
            )

            logger.info("Combining retrievers into an ensemble retriever")
            ensemble_retriever = [bm25_retriever, retriever_vanilla]
            logger.info("Retrievers built successfully.")
            return ensemble_retriever
        except Exception as e:
            logger.error(f"Error building retrievers: {e}")
            raise RuntimeError(f"Error building retrievers: {e}")
        
    def rrf_fusion(
        self,
        retriever_results: List[List[Document]], 
        k: int = 60, 
        top_n: int = 10
    ) -> List[Document]:
        """Reciprocal Rank Fusion for multiple retrievers"""
        scores: Dict[str, float] = {}
        
        # 1. 计算RRF分数
        for results in retriever_results:
            for rank, doc in enumerate(results, 1):
                doc_id = doc.page_content[:50]  # 字符串ID
                scores[doc_id] = scores.get(doc_id, 0) + 1.0 / (k + rank)
        
        # 2. 选top文档ID
        top_docs = heapq.nlargest(top_n, scores, key=scores.get)
        
        # 3. 🔥 修复：seen存doc_id，不是Document！
        unique_docs = []
        seen_ids = set()  # ✅ 字符串set！
        
        for doc_id in top_docs:
            if doc_id in seen_ids:  # ✅ 检查ID！
                continue
                
            for results in retriever_results:
                for doc in results:
                    if doc.page_content[:50] == doc_id:  # ✅ 匹配ID
                        unique_docs.append(doc)
                        seen_ids.add(doc_id)  # ✅ 添加ID到set
                        break
                else:
                    continue
                break
        
        return unique_docs

    def mmr_select(self, query: str, docs: List[Document], k=4, lambda_mult=0.5):
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")

        doc_texts = [d.page_content for d in docs]
        doc_embeddings = embedding.embed_documents(doc_texts)  # List[List[float]]
        query_embedding = embedding.embed_query(query)         # List[float]

        import numpy as np
        query_embedding = np.array(query_embedding)        # List → np.array
        doc_embeddings = np.array(doc_embeddings)          # List → np.array

        selected_indices = maximal_marginal_relevance(
            query_embedding,           # ✅ np.array (shape=(1536,))
            doc_embeddings,            # ✅ np.array (shape=(N, 1536))
            k=k,
            lambda_mult=lambda_mult
        )

        return [docs[i] for i in selected_indices]


    def ensemble_retrieve(self, query: str) -> List[Document]:
            """完整Pipeline: BM25+Embedding → RRF → Cohere Rerank → MMR"""
            logger.info("🔄 开始Ensemble检索...")
            
            # 1. 多检索器检索
            retrievers = self.build_retriever()
            docs = [retriever.invoke(query) for retriever in retrievers]
            logger.info(f"📥 检索到 {sum(len(d) for d in docs)} 个文档")
            
            # 2. RRF融合
            rrf_result = self.rrf_fusion(docs, top_n=8)  # 先多取8个给rerank
            logger.info(f"🔗 RRF融合后: {len(rrf_result)} 个文档")
            
            # 3. 🔥 Cohere Rerank压缩
            reranked_docs = self.cohere_rerank.compress_documents(
                query=query, 
                documents=rrf_result
            )
            logger.info(f"⭐ Cohere Rerank后: {len(reranked_docs)} 个文档")
            
            # 4. MMR多样性选择
            mmr_selected = self.mmr_select(query, reranked_docs, k=4, lambda_mult=0.5)
            logger.info(f"🎯 MMR最终选择: {len(mmr_selected)} 个文档")
            
            return mmr_selected
        