from config.config import (
    CONFLUENCE_URL, CONFLUENCE_USERNAME,
    CONFLUENCE_API_TOKEN, CONFLUENCE_SPACE, PERSIST_DIR
)
from src.loaders.confluence.space_content import ContentProcessorBySpace
from src.vectorstore.vectorstore_manager import VectorStore


class DocumentLoaderService:
    def __init__(self):
        self.content_processor_by_space = ContentProcessorBySpace(
            confluence_url=CONFLUENCE_URL,
            username=CONFLUENCE_USERNAME,
            api_key=CONFLUENCE_API_TOKEN,
            space_key=CONFLUENCE_SPACE
        )

        self.vectorstore_manager = VectorStore(persist_directory=PERSIST_DIR)

    def process_space_documents(self):
        splits = (
            self.content_processor_by_space.load_and_split_content_from_space()
        )
        if not splits:
            return {"message": "No content found in space"}
        page_url = splits[0]["url"]
        self.vectorstore_manager.create_vectorstore(splits, page_url)
        return {"message": "Vectorstore created for space"}

  