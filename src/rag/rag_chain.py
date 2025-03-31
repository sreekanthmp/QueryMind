from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from typing import List, Dict, Any
from config.config import CHAT_MODEL, OPENAI_API_KEY, OPENAI_ORG_ID
from src.rag.prompt_template import PromptManager
from src.vectorstore.load_vectorstore import LoadVectorStore


load_dotenv()

llm = ChatOpenAI(temperature=0.5, api_key=OPENAI_API_KEY,
                 organization=OPENAI_ORG_ID, model=CHAT_MODEL,
                 )


class RAGChain:
    def __init__(self):
        self.vector_loader = LoadVectorStore()

    def _group_documents_by_version(self, retrieved_docs: List[Any]) -> Dict[int, List[Dict[str, Any]]]:
        """Group retrieved documents by version."""
        versioned_responses = {}

        # Flatten nested lists
        flat_docs = [doc for sublist in retrieved_docs for doc in (
            sublist if isinstance(sublist, list) else [sublist])]

        for doc in flat_docs:
            if not hasattr(doc, "metadata"):
                continue

            version = doc.metadata.get("version", "No version available")
            url = doc.metadata.get("url", "No URL available")

            if version not in versioned_responses:
                versioned_responses[version] = []

            versioned_responses[version].append({
                "content": getattr(doc, "page_content",
                                   "No content available"),
                "score": doc.metadata.get("similarity", "N/A"),
                "url": url,
                "metadata": doc.metadata
            })

        return {i: value for i, value in enumerate(versioned_responses.values(), start=1)}

    def _create_prompt_template(self):
        """Retrieve the prompt template from PromptManager."""
        return PromptManager.get_prompt_template()

    def rag_chain(self, question: str) -> Any:
        retriever = self.vector_loader.get_retriever()
        retrieved_docs = retriever.invoke(question)
        status = True
        if not retrieved_docs:
            status = False
            return status, ("failure_message", {})

        versioned_responses = self._group_documents_by_version(retrieved_docs)
        prompt_template = self._create_prompt_template()

        # Format final prompt with history
        final_prompt = prompt_template.format(
            context=retrieved_docs, question=question)

        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt_template
            | llm
            | StrOutputParser()
        )

        response = rag_chain.stream(final_prompt)
        return status, (response, versioned_responses)
