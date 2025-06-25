import pathway as pw
from pathway.xpacks.llm import llms
from RouterAgent import RouterAgent
from ChatAgent import ChatAgent
from ClarifyingAgent import ClarifyingAgent
from SubQueryAgent import SubQueryAgent
from TopicKeywordsAgent import TopicKeywordsAgent
from BaseAgent import create_result_json

class Pipeline:
    def __init__(self, llm: llms.BaseChat, **kwargs):
        self.router_agent = RouterAgent(llm=llm, **kwargs)
        self.chat_agent = ChatAgent(llm=llm, **kwargs)
        self.clarifying_agent = ClarifyingAgent(llm=llm, **kwargs)
        self.sub_query_agent = SubQueryAgent(llm=llm, **kwargs)
        self.topic_keywords_agent = TopicKeywordsAgent(llm=llm, **kwargs)

        self.QuerySchema = self.router_agent.AnswerQuerySchema

    @pw.table_transformer
    def run(self, queries: pw.Table) -> pw.Table:
        # Step 1: Route the query. This is correct.
        routed_queries = queries.with_columns(
            route_destination=self.router_agent.answer_query_table(queries).response
        )
        final_results = routed_queries.with_columns(
            response=pw.if_else(
                pw.this.route_destination == "clarifying_agent",
                self.clarifying_agent.answer_query_table(routed_queries).response,
                pw.if_else(
                    pw.this.route_destination == "sub_query_generating_agent",
                    self.sub_query_agent.answer_query_table(routed_queries).response,
                    pw.if_else(
                        pw.this.route_destination == "topic_and_keywords_agent",
                        self.topic_keywords_agent.answer_query_table(routed_queries).response,
                        self.chat_agent.answer_query_table(routed_queries).response,
                    ),
                ))
        )        
        return final_results.select(
            result=create_result_json(pw.this.query, pw.this.response)
        )

class PipelineServer:
    def __init__(self, host: str, port: int, pipeline: Pipeline, **rest_kwargs):
        self.webserver = pw.io.http.PathwayWebserver(host=host, port=port)
        queries, writer = pw.io.http.rest_connector(
            webserver=self.webserver,
            route="/v1/chat",
            schema=pipeline.QuerySchema,
            autocommit_duration_ms=50,
            delete_completed_queries=True,
            **rest_kwargs
        )
        writer(pipeline.run(queries))

    def run(self):
        pw.run()

if __name__ == "__main__":

    HOST = "0.0.0.0"
    PORT = 8000
    
    chat_llm = llms.LiteLLMChat(
        model="groq/llama3-8b-8192"
    )

    pipeline = Pipeline(llm=chat_llm)
    server = PipelineServer(host=HOST, port=PORT, pipeline=pipeline)
    server.run()