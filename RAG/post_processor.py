import hashlib
from typing import List
from langchain_core.documents import Document
from RAG.retriever_utils import content_hash
class PostProcessor:
    def __init__(self, raw_retrieved_docs: List[Document]):
        self.docs = raw_retrieved_docs
        self.unique_doc = []

    def dedup_by_content(self):
        seen = set()

        for doc in self.docs:
            h = content_hash(doc.page_content)
            if h not in seen:
                seen.add(h)
                self.unique_doc.append(doc)
        return self.unique_doc
    
    def context_format(self) -> str:
        context = ""
        for doc in self.unique_doc:
            context += f"""
        Source: {doc.metadata.get('source')}
        Relevant score:
        {doc.metadata.get('relevance_score')}
        Relevant Content:
        {doc.page_content}
        ------------------------------------
        \n\n
        """
        return context