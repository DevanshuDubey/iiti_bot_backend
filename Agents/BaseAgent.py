import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

import pathway as pw
from pathway.xpacks.llm import llms, servers
from pathway.xpacks.llm.question_answering import BaseQuestionAnswerer

@pw.udf
def create_result_json(query: str, response: str) -> pw.Json:
    return pw.Json({"query": query, "response": response})

class BaseAgent(BaseQuestionAnswerer):
    def __init__(
        self,
        llm:llms.BaseChat,
        *,
        default_llm_name: str | None = None,
        prompt_template: str = None,
    ):
        self.llm = llm
        if default_llm_name is None:
            default_llm_name = llm.model

        self.prompt_template = prompt_template
        self._init_schemas(default_llm_name)
    
    def _init_schemas(self, default_llm_name: str | None = None) -> None:
        class BaseQuerySchema(pw.Schema):
            query: str
            model: str | None = pw.column_definition(default_value=default_llm_name)
        self.AnswerQuerySchema = BaseQuerySchema
        self.RetrieveQuerySchema = pw.Schema
        self.StatisticsQuerySchema = pw.Schema
        self.InputsQuerySchema = pw.Schema

    @pw.table_transformer 
    def answer_query_table(self, queries: pw.Table) -> pw.Table:
        """
        Returns answer as pw.Table"""
        
        results = queries.with_columns(
            prompt=pw.apply(
                lambda query_string: self.prompt_template.format(query=query_string),
                pw.this.query,
            )
        )

        
        results = results.with_columns(
            response=self.llm(llms.prompt_chat_single_qa(pw.this.prompt), model=pw.this.model)
        )
        
        return results
    
    @pw.table_transformer
    def answer_query(self, queries: pw.Table) -> pw.Table:
        """
        Returns answer as Json"""
        return self.answer_query_table(queries).select(
            result=create_result_json(pw.this.query, pw.this.response)
        )



class CustomServer(servers.BaseRestServer):
    def __init__(
        self,
        host: str,
        port: int,
        router_agent_answerer: "BaseAgent",
        **rest_kwargs,
    ):
        super().__init__(host, port, **rest_kwargs)
        self.serve(
            route="/v1/chat",
            schema=router_agent_answerer.AnswerQuerySchema,
            handler=router_agent_answerer.answer_query,
            **rest_kwargs,
        )
