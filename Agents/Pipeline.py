import os
import pathway as pw
from pathway.xpacks.llm import llms, servers

# Import agent classes and UDFs from your files
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
        routed_queries = queries.with_columns(
            route_destination=self.router_agent.answer_query_table(queries).response
        )

        clarifying_path = routed_queries.filter(pw.this.route_destination == "clarifying_agent")
        clarifying_results = self.clarifying_agent.answer_query_table(clarifying_path).select(
            pw.this.query, pw.this.response
        )

        sub_query_path = routed_queries.filter(pw.this.route_destination == "sub_query_generating_agent")
        sub_query_results = self.sub_query_agent.answer_query_table(sub_query_path).select(
            pw.this.query, pw.this.response
        )

        topic_path = routed_queries.filter(pw.this.route_destination == "topic_and_keywords_agent")
        topic_results = self.topic_keywords_agent.answer_query_table(topic_path).select(
            pw.this.query, pw.this.response
        )

        chat_path = routed_queries.filter(
            ~(
                (pw.this.route_destination == "clarifying_agent") |
                (pw.this.route_destination == "sub_query_generating_agent") |
                (pw.this.route_destination == "topic_and_keywords_agent")
            )
        )
        chat_results = self.chat_agent.answer_query_table(chat_path).select(
            pw.this.query, pw.this.response
        )
        final_results = pw.Table.concat_reindex(
            
            clarifying_results,
            sub_query_results,
            topic_results,
            chat_results
        )
        
        return final_results


class CustomServer(servers.BaseRestServer):
    def __init__(
        self,
        host: str,
        port: int,
        router_agent_answerer: "Pipeline",
        **rest_kwargs,
    ):
        super().__init__(host, port, **rest_kwargs)
        self.serve(
            route="/v1/chat",
            schema=router_agent_answerer.QuerySchema,
            handler=router_agent_answerer.run,
            **rest_kwargs,
        )

bot = llms.LiteLLMChat(
    model="groq/llama3-8b-8192"
)
server = CustomServer(host="0.0.0.0",port=8000,router_agent_answerer=Pipeline(bot))
server.run()
