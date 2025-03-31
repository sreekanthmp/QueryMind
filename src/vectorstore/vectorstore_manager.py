import logging
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

from config.config import EMBEDDING_MODEL


class VectorStore:
    def __init__(self, persist_directory: str):
        self.persist_directory = persist_directory
        self.vectorstore = Chroma(
            embedding_function=OpenAIEmbeddings(
                model=EMBEDDING_MODEL),
            persist_directory=self.persist_directory,
        )

    def create_vectorstore(self, splits: list, page_url: str):
        texts = [split["content"] for split in splits]
        metadatas = [{"url": split["url"], "title": split["title"],
                      "version": split["version"]} for split in splits]

        # Handling duplicates based on URL and title
        try:
            existing_docs = self.vectorstore._collection.get(
                include=["metadatas"])
            existing_ids = [
                doc_id for doc_id, metadata in zip(existing_docs["ids"],
                                                   existing_docs["metadatas"])
                if metadata.get("url") == (
                    page_url and metadata.get("title") == splits[0]["title"]
                )
            ]

            # Remove existing entries for this page
            if existing_ids:
                logging.info(
                    f"Removing {len(existing_ids)} existing splits for "
                    f"{page_url}")
                self.vectorstore._collection.delete(where={"url": page_url})
        except Exception as e:
            logging.warning(f"Could not retrieve existing data: {e}")

        # Add new splits to the Vectorstore
        self.vectorstore.add_texts(texts=texts, metadatas=metadatas)
        logging.info(
            f"Vectorstore updated and persisted at {self.persist_directory}")

        return self.vectorstore

    