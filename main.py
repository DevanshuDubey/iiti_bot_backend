import warnings
import os
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

import Agents.BaseAgent as BaseAgent
import pathway as pw
from pathway.xpacks.llm.embedders import SentenceTransformerEmbedder
from pathway.xpacks.llm.parsers import UnstructuredParser
from pathway.xpacks.llm.splitters import RecursiveSplitter
from pathway.xpacks.llm.rerankers import CrossEncoderReranker
from pathway.xpacks.llm import llms
from pathway.xpacks.llm.document_store import DocumentStore
from pathway.stdlib.indexing import HybridIndexFactory, BruteForceKnnFactory, TantivyBM25Factory
from pathway.xpacks.llm.question_answering import AdaptiveRAGQuestionAnswerer, BaseQuestionAnswerer
from pydantic import BaseModel, ConfigDict
from pathway.xpacks.llm.servers import BaseRestServer, QARestServer

source = pw.io.fs.read(
    path="./data",
    mode="static",
    format="binary",
    with_metadata=True
)

embedding = SentenceTransformerEmbedder(
    model="Qwen/Qwen3-Embedding-0.6B"
)

parser = UnstructuredParser()

splitter = RecursiveSplitter(chunk_size=300,chunk_overlap=50)

retriever = HybridIndexFactory(
    retriever_factories=[BruteForceKnnFactory(embedder=embedding),TantivyBM25Factory(ram_budget=500*1024*1024,in_memory_index=False)]
)

document_store = DocumentStore(
    docs=source,
    parser=parser,
    splitter=splitter,
    retriever_factory=retriever,   
)


chatmodel  = llms.LiteLLMChat(
    model="groq/llama3-8b-8192",
)

# chatmodel  = llms.LiteLLMChat(
#     model="gemini/gemini-2.0-flash-lite"
# )

# bot = BaseAgent.RouterAgentAnswerer(
#     llm=chatmodel
# )


bot = AdaptiveRAGQuestionAnswerer(
    indexer=document_store,
    llm=chatmodel,
    no_answer_string="XXXXXXNONONONXXXXXXX",
    n_starting_documents=4,
    factor=1,
    max_iterations=3,
)


host: str = "0.0.0.0"
port: int = 8000
server = QARestServer(
    host=host,
    port=port,
    router_agent_answerer=bot
)

server.run()