import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

import pathway as pw
from pathway.xpacks.llm import llms, servers

from RouterAgent import RouterAgent
from ChatAgent import ChatAgent
from ClarifyingAgent import ClarifyingAgent
from SubQueryAgent import SubQueryAgent
from TopicKeywordsAgent import TopicKeywordsAgent

@pw.udf
def create_json(query: str, route: str, response: str) -> pw.Json:
    return pw.Json({"query": query,"route": route, "response": response})


class Pipeline:
    def __init__(self, llm: llms.BaseChat, **kwargs):
        self.llm = llm
        self.router_agent = RouterAgent(llm=llm, **kwargs)
        
        self.agent_map = {
            "clarifying_agent": ClarifyingAgent(llm=llm, **kwargs),
            "sub_query_generating_agent": SubQueryAgent(llm=llm, **kwargs),
            "topic_and_keywords_agent": TopicKeywordsAgent(llm=llm, **kwargs),
            "chat_agent": ChatAgent(llm=llm, **kwargs),
        }
        self.QuerySchema = self.router_agent.AnswerQuerySchema

    @pw.table_transformer
    def run(self, queries: pw.Table) -> pw.Table:

        routed_queries = queries.with_columns(
            route_destination=self.router_agent.answer_query_table(queries).response
        )

        @pw.udf
        def generate_prompt(query: str, route: str) -> str:
            agent = self.agent_map.get(route, self.agent_map["chat_agent"])
            return agent.prompt_template.format(query=query)

        prompted_queries = routed_queries.with_columns(
            prompt=generate_prompt(pw.this.query, pw.this.route_destination)
        )
        
        final_results = prompted_queries.with_columns(
            response=self.llm(llms.prompt_chat_single_qa(pw.this.prompt), model=pw.this.model)
        )
        x = final_results.select(
            result=create_json(pw.this.query,routed_queries.route_destination, pw.this.response)
        )
        
        return x


class CustomServer(servers.BaseRestServer):
    def __init__(
        self,
        host: str,
        port: int,
        pipeline: "Pipeline",
        **rest_kwargs,
    ):
        super().__init__(host, port, **rest_kwargs)
        self.serve(
            route="/v1/chat",
            schema=pipeline.QuerySchema,
            handler=pipeline.run,
            **rest_kwargs,
        )

bot = llms.LiteLLMChat(
    model="groq/llama3-70b-8192"
)
server = CustomServer(host="0.0.0.0", port=8000, pipeline=Pipeline(bot))
server.run()