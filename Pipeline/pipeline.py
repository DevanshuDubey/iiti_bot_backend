import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)
import os
import pathway as pw
from pathway.xpacks.llm import llms
from typing import List
from Agents.RouterAgent import RouterAgent
from Agents.ChatAgent import ChatAgent
from Agents.ClarifyingAgent import ClarifyingAgent
from Agents.SubQueryAgent import SubQueryAgent
from .subPipeline import sub_pipeline
 
os.environ["GROQ_API_KEY"]="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"

@pw.udf
def create_final_json(query: str, route: str, llm_response: str, model) -> pw.Json:
    subqueries: List[str] | None = None
    response_value: str
    subqueries = None
    response_value = llm_response
    if route == "clarifying_agent" or route == "chat_agent":
        return pw.Json({
         "check" : "hiiiiiiiiii",
        "route" : route, 
        "status": "success",
        "text": response_value
        })

    subqueries = [q.strip() for q in llm_response.split("<SBQ>") if q.strip()]
    response , score , feedback , iteration_counter , doc_string , query_string = sub_pipeline(query, subqueries, 4)
     
    return pw.Json({
        "check" : "hiiiiiiiiii",
        "status": "success",
        "query_string": query_string,
        "doc_string": doc_string,
        "text": response["text"],
        "snippetText": response["source_snippet"],
        "iteration_counter": iteration_counter,
        "score": score,
        "feedback": feedback
        })

     
class Pipeline:
    def __init__(self, llm: llms.BaseChat, **kwargs):
        self.llm = llm
        self.router_agent = RouterAgent(llm=llm, **kwargs)
        
        self.agent_map = {
            "clarifying_agent": ClarifyingAgent(llm=llm, **kwargs),
            "chat_agent": ChatAgent(llm=llm, **kwargs),
            "sub_query_generating_agent": SubQueryAgent(llm=llm, **kwargs)       
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
            llm_response=self.llm(llms.prompt_chat_single_qa(pw.this.prompt), model=pw.this.model)
        )

        output = final_results.select(
            result=create_final_json(pw.this.query, pw.this.route_destination, pw.this.llm_response, pw.this.model)  
        )

        return output
        
