from langchain_text_splitters import MarkdownHeaderTextSplitter
from docling.document_converter import DocumentConverter
from pathlib import Path
from markdown import markdown
from docling_core.types import DoclingDocument
import logging

logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, headers_to_split_on: list[str], pdf_file_pth: str="/Users/george/ai-projects/MultiAgenticRAG_Rep/papers/2310.08560v2.pdf"):
        self.headers_to_split_on = [
            ("#", "Header 1"),
            ("##", "Header 2"),
            ("###", "Header 3")
        ]
        self.pdf_file_pth = pdf_file_pth
    
    def _convert_to_doclingDocument(self) -> DoclingDocument:
        source = self.pdf_file_pth
        converter = DocumentConverter()
        doc = converter.convert(source=source).document
        return doc

    def _convert_to_html(self, doc: DoclingDocument):
        html = doc.export_to_html()
        with open("output.html", "w", encoding="utf-8") as f:
            f.write(html)
        print("👍 Saved as html file")

    def _convert_to_md(self, doc: DoclingDocument):
        md = doc.export_to_markdown()
        return md

    def process_split(self):
        try:
            logger.info("Starting document processing.")
            markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=self.headers_to_split_on)
            doc = self._convert_to_doclingDocument()
            md_doc = self._convert_to_md(doc)
            chunked_doc = markdown_splitter.split_text(md_doc)
            print(f"\n👌 Split into {len(chunked_doc)} chunks")
            print(f"\nThese are the chunked doc: {type(chunked_doc[0])}")
            return chunked_doc
        except Exception as e:
            logger.error(f"Error processing document: {e}")
            raise RuntimeError(f"Error processing document: {e}")