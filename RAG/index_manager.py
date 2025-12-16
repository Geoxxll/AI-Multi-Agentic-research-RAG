from RAG.document_processor import DocumentProcessor
from RAG.index_builder import IndexBuilder

class GlobalIndexManager:
    _vectorstore = None
    _chunked_doc = None

    @classmethod
    def get_vectorstore(cls, headers_to_split_on, file_pth):
        if cls._vectorstore is not None:
            return cls._vectorstore, cls._chunked_doc

        print("🧠 Building vectorstore (first time only)...")

        doc_processor = DocumentProcessor(headers_to_split_on=headers_to_split_on)
        cls._chunked_doc = doc_processor.process_split()

        index_builder = IndexBuilder(
            chunked_doc=cls._chunked_doc,
            collection_name="test",
            persist_directory="./RAG",
            load_documents=True
        )

        cls._vectorstore = index_builder.build_vectorstore()
        return cls._vectorstore, cls._chunked_doc
