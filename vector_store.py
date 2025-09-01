import os
from typing import List, Dict, Any
from langchain_postgres.vectorstores import PGVector
from langchain.docstore.document import Document
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

load_dotenv()

DBNAME = os.getenv("DB_NAME")
DBUSER = os.getenv("DB_USER")
DBPW = os.getenv("DB_PASSWORD")
DBHOST = os.getenv("DB_HOST")
DBPORT = os.getenv("DB_PORT")

PGVECTOR_CONNECTION_STRING = f"postgresql+psycopg2://{DBUSER}:{DBPW}@{DBHOST}:{DBPORT}/{DBNAME}"

class VectorStorePGVector:
    def __init__(self, collection_name: str = "book_embeddings"):
        """
        Initializes the vector store connection.

        Args:
            collection_name (str): Name of the pgvector collection.
        """
        self.collection_name = collection_name
        self.db_connection_string = PGVECTOR_CONNECTION_STRING

        if not self.db_connection_string:
            raise ValueError("Database connection string is missing in environment variables.")

        self.embeddings_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        self.vector_store = PGVector(
            connection=self.db_connection_string,
            collection_name=self.collection_name,
            embeddings=self.embeddings_model,
            use_jsonb=True
        )

    def store_docs_to_collection(self, book_id: str, doc_chunks: List[Dict[str, Any]]):
        """
        Stores chapter-wise document chunks with embeddings into pgvector.

        Args:
            book_id (str): Unique book identifier.
            doc_chunks (List[Dict[str, Any]]): List of chapter-wise chunks.
        """
        docs = [
            Document(
                page_content=chunk["text"],  
                metadata={"book_id": book_id, "chapter_name": chunk["chapter_name"]}
            )
            for chunk in doc_chunks
        ]

        self.vector_store.add_documents(docs)
        return {"status": "Documents stored successfully", "book_id": book_id}

    def check_if_record_exist(self, book_id: str) -> bool:
        """
        Checks if embeddings for a given book_id exist in the vector store.

        Args:
            book_id (str): Unique book identifier.

        Returns:
            bool: True if records exist, False otherwise.
        """
        query_text = "Sample text"
        embedding_vector = self.embeddings_model.embed_query(query_text)

        results = self.vector_store.similarity_search_by_vector(embedding=embedding_vector, k=1)

        return any(doc.metadata.get("book_id") == book_id for doc in results)

    def retrieve_relevant_chapter_chunks(self, book_id: str, chapter_name: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Retrieves the most relevant text chunks for a given chapter.

        Args:
            book_id (str): Unique book identifier.
            chapter_name (str): Chapter name for retrieving text.
            k (int): Number of chunks to retrieve.

        Returns:
            List[Dict[str, Any]]: Retrieved chapter text chunks.
        """
        query_text = f"Chapter: {chapter_name}"
        results = self.vector_store.similarity_search(query_text, k=k)

        return [
            {"chapter_name": doc.metadata["chapter_name"], "text": doc.page_content}
            for doc in results
            if doc.metadata.get("book_id") == book_id
        ]
 