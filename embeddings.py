import os
import re
from typing import Dict, List, Any
from langchain_community.document_loaders import PyPDFLoader
from upload.vector_store import VectorStorePGVector


class EmbeddingGenerator:
    def __init__(self, collection_name: str = "book_embeddings"):
        """
        Initializes the embedding generator.
        
        Args:
            collection_name (str): Name of the pgvector collection.
        """
        self.vector_store = VectorStorePGVector(collection_name)

    def extract_and_chunk_text(self, file_path: str) -> Dict[str, str]:
        """
        Extracts text from a PDF and organizes it into chapters.

        Args:
            file_path (str): Path to the uploaded PDF file.

        Returns:
            Dict[str, str]: Dictionary where keys are chapter names, and values are text content.
        """
        loader = PyPDFLoader(file_path)
        pages = loader.load()
        full_text = "\n".join([page.page_content for page in pages])
        full_text = full_text.replace("\x00", "")

        # Regex to match chapter titles in format "CHAPTER X:- Chapter Title"
        pattern = r'CHAPTER\s+\d+:-\s+(.*?)\s+\d+'
        matches = list(re.finditer(pattern, full_text))

        if not matches:
            raise ValueError("No chapter titles found. Ensure chapters are formatted correctly.")

        chapters = {}
        for i, match in enumerate(matches):
            chapter_name = match.group(1).strip()
            print("chapter name extracted", chapter_name)
            start_index = match.end()
            end_index = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
            chapter_text = full_text[start_index:end_index].strip()
            chapter_text = chapter_text.replace("\x00", "")

            if chapter_text:
                chapters[chapter_name] = chapter_text

        return chapters

    def create_embeddings(self, file_path: str, book_id: str):
        """
        Extracts text, organizes it into chapters, checks if embeddings exist, and stores them.

        Args:
            file_path (str): Path to the uploaded PDF file.
            book_id (str): Unique identifier for the book.
        """
        # Check if embeddings already exist
        if self.vector_store.check_if_record_exist(book_id):
            return {"status": "Embeddings already exist for this book", "book_id": book_id}

        chapter_texts = self.extract_and_chunk_text(file_path)

        doc_chunks = [
            {"chapter_name": chapter_name, "text": text}
            for chapter_name, text in chapter_texts.items()
        ]

        # Store embeddings in pgvector
        self.vector_store.store_docs_to_collection(book_id, doc_chunks)

        return {"status": "Embeddings stored successfully", "book_id": book_id}
