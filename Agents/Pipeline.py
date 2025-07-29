import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
import os
os.environ["GROQ_API_KEY"]="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"
import pathway as pw
from pathway.xpacks.llm import llms, servers
from typing import List
from RouterAgent import RouterAgent
from ChatAgent import ChatAgent
from ClarifyingAgent import ClarifyingAgent
from SubQueryAgent import SubQueryAgent
from TopicKeywordsAgent import TopicKeywordsAgent
from AnswerGeneratingAgent import AnswerGeneratingAgent
import requests
from Abhinav.further import further_pipeline
# from CritiqueAgent import CritiqueAgent
# @pw.udf
# def access()


bot = llms.LiteLLMChat(
    model="groq/llama3-70b-8192"
)

@pw.udf
def create_final_json(query: str, route: str, llm_response: str, model) -> pw.Json:
    subqueries: List[str] | None = None
    response_value: str
    subqueries = None
    response_value = llm_response
    if route == "clarifying_agent" or route == "chat_agent":
        return pw.Json({
        "status": "success",
        "text": response_value
        })

    # if route == "sub_query_generating_agent":
    subqueries = [q.strip() for q in llm_response.split("<SBQ>") if q.strip()]
    response = further_pipeline(query, subqueries, 4)
    # prompt = AnswerGeneratingAgent.prompt_template

    # response = bot(prompt.format(query=query_string, docs=doc_string))

    # response is a dict/json - take out values and return the final response  
    

    # return pw.Json({
    #     "success":"true",
    #     "response": bot(llms.prompt_chat_single_qa("Answer my quyestuion"), model=model)
    # })
        
    return pw.Json({
        "check" : "hiiiiiiiiii"
        # "status": "success",
        # "text": response["text"],
        # "source_snippet": response["source_snippet"]
        })

    # return pw.Json({
    #     "query": query,
    #     "route": route,
    #     "subqueries": subqueries,
    #     "response": response_value
    # })


class Pipeline:
    def __init__(self, llm: llms.BaseChat, **kwargs):
        self.llm = llm
        self.router_agent = RouterAgent(llm=llm, **kwargs)
        
        self.agent_map = {
            "clarifying_agent": ClarifyingAgent(llm=llm, **kwargs),
            "sub_query_generating_agent": SubQueryAgent(llm=llm, **kwargs),
            # "topic_and_keywords_agent": TopicKeywordsAgent(llm=llm, **kwargs),
            "chat_agent": ChatAgent(llm=llm, **kwargs),
            # "aga_agent": AnswerGeneratingAgent(llm=llm, **kwargs),
            # "critique_agent": CritiqueAgent(llm=llm, **kwargs),
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
        
        @pw.udf()
        def smth(route : str):
            if route == 'chat_agent':
                return route
            else:
                return "smth"
        
        prompted_queries = routed_queries.with_columns(
            prompt=generate_prompt(pw.this.query, pw.this.route_destination)
        )
        
        final_results = prompted_queries.with_columns(
            llm_response=self.llm(llms.prompt_chat_single_qa(pw.this.prompt), model=pw.this.model)
        )

        output = final_results.select(
            result=create_final_json(pw.this.query, pw.this.route_destination, pw.this.llm_response, pw.this.model)  
        )

        # result1 = create_final_json(final_results.query, final_results.route_destination, final_results.llm_response)
        # print("*********************************************************************************************************************88")
        
        # dict = pw.debug.table_to_dicts(output)
        # print(dict)
        # @pw.udf
        # def return_output(routed_queries, output):

        #     pw.output.result = pw.routed_queries.route_destination + pw.output.result
        #     # output= output.with_columns(
        #     #     # result = pw.this.result + pw.routed_queries.route
        #     #     result = output.select(result=pw.this.result).concat(routed_queries.select(route=pw.this.route))
        #     # )
        #     output = output.with_columns(routed_queries.select(pw.this.route_destination))
        #     pw.udf()
        #     def fun(route:str):
        #         if route == "chat_agent":
        #             return 

        #     output = output.with_columns()
        #     return output
        # return return_output
        return output
        






## cors
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

 
server = CustomServer(host="0.0.0.0", port=8001, pipeline=Pipeline(bot),
    # allow_origin="*",
    # allow_methods=["POST", "GET", "OPTIONS"],
    # allow_headers=["Content-Type", "Authorization"]                  
    )
server.run()