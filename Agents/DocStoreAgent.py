import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

import pathway as pw
from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder
from pathway.xpacks.llm import splitters
from pathway.xpacks.llm.parsers import UnstructuredParser
from pathway.xpacks.llm.llms import LiteLLMChat
from pathway.xpacks.llm.document_store import DocumentStore
from pathway.stdlib.indexing import BruteForceKnnFactory
from pathway.xpacks.llm.question_answering import AdaptiveRAGQuestionAnswerer
from pathway.xpacks.llm import servers

class DocStoreAgent(AdaptiveRAGQuestionAnswerer):
    def __init__(self, **kwargs):
        llm = LiteLLMChat(model="groq/llama3-70b-8192")
        embedder = SentenceTransformerEmbedder(model="all-MiniLM-L6-v2")
        parser = UnstructuredParser()
        splitter = splitters.RecursiveSplitter(
            chunk_size=400,
            chunk_overlap=100,
        )
        docs = pw.io.fs.read(
            "./data",
            format="binary",
            mode="streaming",
            with_metadata=True,
        )
        document_store = DocumentStore(
            docs=docs,
            parser=parser,
            splitter=splitter,
            retriever_factory=BruteForceKnnFactory(embedder=embedder),
        )

        super().__init__(
            llm=llm,
            indexer=document_store,
            max_iterations=3,
            **kwargs,
        )
        self._init_custom_schemas()

    def _init_custom_schemas(self) -> None:
        #----------QUERY SCHEMA-------------------
        class CustomQuerySchema(pw.Schema):
            query: str
            k: int = pw.column_definition(default_value=3)


        self.CustomQuerySchema = CustomQuerySchema
    
    @pw.table_transformer
    def fetch_docs(self, query: pw.Table) -> pw.Table:
        query_with_full_schema = query.with_columns(
            metadata_filter=None,
            filepath_globpattern=None
        )
        return self.retrieve(query_with_full_schema)


# class CustomServer(servers.BaseRestServer):
#     def __init__(
#         self,
#         host: str,
#         port: int,
#         answerer: "DocStoreAgent",
#         **rest_kwargs,
#     ):
#         super().__init__(host, port, **rest_kwargs)
#         self.serve(
#             route="/v1/fetch_documents",
#             schema=answerer.CustomQuerySchema,
#             handler=answerer.fetch_docs,
#             **rest_kwargs,
#         )

# bot = DocStoreAgent()
# server = CustomServer(
#     host="0.0.0.0",
#     port=8002,
#     answerer=bot,
# )
# server.run()