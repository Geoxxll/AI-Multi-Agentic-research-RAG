import os
from typing import List
from langchain_openai import OpenAIEmbeddings
import logging
from langchain_chroma import Chroma
logger = logging.getLogger(__name__)

class IndexBuilder:
    """
    Builds a vector-based
    """
    def __init__(self, chunked_doc: List[str], collection_name: str, persist_directory: str, load_documents: bool):
        self.chunked_doc = chunked_doc
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.load_documents = load_documents

    def build_vectorstore(self):
        """
        Initializes the Chroma vectorstore with the provided documents and embeddings
        """
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        try:
            logger.info("Building VectorStore")
            if not os.path.exists(self.persist_directory):
                logger.info("🧠 Detect persist_directory not exist...CREATING...")
                self.vectorstore = Chroma.from_documents(
                    documents=self.chunked_doc,
                    collection_name=self.collection_name,
                    embedding=embeddings,
                    persist_directory=self.persist_directory
                )
            else:
                logger.info("📦 Detect persisten_directory exist...LOADING...")
                self.vectorstore = Chroma(
                    persist_directory=self.persist_directory,
                    collection_name=self.collection_name,
                    embedding_function=embeddings
                )
            logger.info("🔥 Vectorstore built/load successfully.")
            return self.vectorstore
        except Exception as e:
            logger.error(f"Error building vectorstore: {e}")
            raise RuntimeError(f"Error building vectorsrore: {e}")
            
        