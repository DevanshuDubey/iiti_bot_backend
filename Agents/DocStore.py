import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

from pathway.xpacks.llm.document_store import DocumentStore

#making a dummy document store agent for experimentation
class DocumentStoreAgent:
    def __init__(self, document_store: DocumentStore):
        self.document_store = document_store

    def retrieve(self, query: str):
        return self.document_store.retrieve_query(query)
