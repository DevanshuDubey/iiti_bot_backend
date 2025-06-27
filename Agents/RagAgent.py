from BaseAgent import BaseAgent, CustomServer
import pathway as pw
from pathway.xpacks.llm import llms
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

#dummy rag for experimentation
class RagAgent(BaseAgent):
    PROMPT_TEMPLATE = """
    You are an expert question-answering agent. Your task is to answer the user's question using the provided context.
    
    ---
    [CONTEXT]
    {context}
    
    ---
    [USER QUERY]
    User Query: "{query}"
    Answer:"""
    def __init__(self, llm: llms.BaseChat, **kwargs):
        super().__init__(
            llm=llm,
            prompt_template=self.PROMPT_TEMPLATE,
            **kwargs,
        )
    
    @pw.table_transformer
    def answer_query(self, queries: pw.Table, docs : pw.Table) -> pw.Table:
        prompt = self.prompt_template.format(
            query=queries.query,
            context=docs.result["text"]
        )
        
        response= self.llm(llms.prompt_chat_single_qa(prompt), model=self.default_llm_name)
        
        queries = queries.with_columns(
            response=response
        )
        
        return queries
