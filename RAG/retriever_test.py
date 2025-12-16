from RAG.document_processor import DocumentProcessor
from RAG.index_builder import IndexBuilder
from RAG.retriever_builder import Retrievers

def main():
    headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3")
        ]
    doc_processor = DocumentProcessor(headers_to_split_on=headers_to_split_on)
    chunked_doc = doc_processor.process_split()
    index_builder = IndexBuilder(chunked_doc=chunked_doc, collection_name="test", persist_directory="./RAG", load_documents=True)
    vectorstore = index_builder.build_vectorstore()
    retrievers = Retrievers(chunked_doc=chunked_doc, vectorstore=vectorstore)
    query = "What architecture does MemCPT have?"
    final_docs = retrievers.ensemble_retrieve(query=query)
    print(f"\n✅ Here are the final selected doc from RAG pipeline: {final_docs}")
if __name__ == "__main__":
    main()