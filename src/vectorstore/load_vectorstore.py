import logging
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from config.config import EMBEDDING_MODEL
from typing import List, Dict, Any


class LoadVectorStore:
    def __init__(self):
        self.vectorstore = self._initialize_vectorstore()

    def _initialize_vectorstore(self):
        """Initialize and return the vectorstore."""
        persist_directory = "./chroma_db_search"
        return Chroma(
            persist_directory=persist_directory,
            embedding_function=OpenAIEmbeddings(model=EMBEDDING_MODEL),
        )

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Retrieve all documents from the vectorstore with their metadata."""
        try:
            all_docs = self.vectorstore._collection.get(include=["metadatas"])
            return all_docs["metadatas"]
        except Exception as e:
            logging.error(str(e))
            return []

    def get_retriever(self):
        """Create and return a retriever from the vectorstore."""
        return self.vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={'score_threshold': 0.0}
        )
