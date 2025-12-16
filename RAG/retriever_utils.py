from langchain_community.vectorstores.utils import maximal_marginal_relevance
import heapq
import hashlib
from typing import List, Dict
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from RAG.document_processor import DocumentProcessor
from RAG.index_builder import IndexBuilder
from RAG.index_manager import GlobalIndexManager
from RAG.retriever_builder import Retrievers

def retrieve(headers_to_split_on, query, file_pth):
    vectorstore, chunked_doc = GlobalIndexManager.get_vectorstore(
        headers_to_split_on=headers_to_split_on,
        file_pth=file_pth
    )

    retrievers = Retrievers(
        chunked_doc=chunked_doc,
        vectorstore=vectorstore
    )

    final_docs = retrievers.ensemble_retrieve(query=query)
    print(f"\n✅ There are {len(final_docs)} documents selected from RAG pipeline....")
    return final_docs

def rrf_fusion(
    retriever_results: List[List[Document]], 
    k: int = 60, 
    top_n: int = 10
) -> List[Document]:
    """Reciprocal Rank Fusion for multiple retrievers"""
    scores: Dict[str, float] = {}
    
    for results in retriever_results:
        for rank, doc in enumerate(results, 1):
            doc_id = getattr(doc, 'page_content', str(doc))[:50]  # Use content hash
            if doc_id not in scores:
                scores[doc_id] = 0
            scores[doc_id] += 1.0 / (k + rank)
    
    # Get top documents by RRF score
    top_docs = heapq.nlargest(top_n, scores, key=scores.get)
    
    # Return unique documents
    unique_docs = []
    seen = set()
    for doc_id in top_docs:
        for results in retriever_results:
            for doc in results:
                if getattr(doc, 'page_content', str(doc))[:50] == doc_id and doc not in seen:
                    unique_docs.append(doc)
                    seen.add(doc)
                    break
    
    return unique_docs


def mmr_select(
            query: str,
            docs: str,
            k: int=4,
            lambda_mult: float=0.5
    ):
        embedding = OpenAIEmbeddings(model="text-embedding-3-small")
        doc_texts = [d.page_content for d in docs]
        doc_embeddings = embedding.embed_documents(doc_texts)
        query_embedding = embedding.embed_query(query)
        
        selected_indices = maximal_marginal_relevance(
            query_embedding,
            doc_embeddings,
            k=k,
            lambda_mult=lambda_mult
        )
        
        return [docs[i] for i in selected_indices]


def ensemble_retrieve(retrievers, query: str) -> List[Document]:
    docs = [retriever.invoke(query) for retriever in retrievers]
    rff_result = rrf_fusion(docs, top_n=4)
    mmr_selected = mmr_select(query, rff_result)
    return mmr_selected

def content_hash(text: str) -> str:
    return hashlib.md5(text.strip().encode("utf-8")).hexdigest()