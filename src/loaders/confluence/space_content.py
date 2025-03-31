import logging
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import ConfluenceLoader


class ContentProcessorBySpace:
    def __init__(self, confluence_url, username, api_key, space_key):
        self.confluence_url = confluence_url
        self.username = username
        self.api_key = api_key
        self.space_key = space_key
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=512, chunk_overlap=10
        )

    def fetch_all_pages(self) -> list:
        """
        Fetch all pages in the Confluence space.
        """
        confluence_loader = ConfluenceLoader(
            url=self.confluence_url,
            username=self.username,
            api_key=self.api_key,
            space_key=self.space_key
        )
        return confluence_loader.load()

    def load_and_split_content_from_space(self) -> list:
        """
        Fetch all pages, load, and split their content into smaller chunks.
        """
        all_pages = self.fetch_all_pages()
        split_results = []

        for doc in all_pages:
            page_id = doc.metadata.get('id')
            page_title = doc.metadata.get('title', 'Untitled Page')
            page_version = doc.metadata.get('version', 'Unknown')
            page_url = (
                f"{self.confluence_url}/spaces/{self.space_key}"
                f"/pages/{page_id}"
            )
            # Split content
            splits = self.text_splitter.split_text(doc.page_content)
            # Store metadata for each chunk
            for split in splits:
                split_results.append({
                    "url": page_url,
                    "title": page_title,
                    "version": page_version,
                    "content": split
                })

        logging.info(f"Total splits for space: {len(split_results)}")
        return split_results
