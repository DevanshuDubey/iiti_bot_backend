from BaseAgent import BaseAgent, CustomServer
from pathway.xpacks.llm import llms
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

class SubQueryAgent(BaseAgent):
    PROMPT_TEMPLATE = """
    You are an expert in generating sub-queries for a given query. Your task is to generate sub-queries for a given query. Respond with ONLY the sub-queries separated by commas.
    
    ----
    [EXAMPLES]
    Query: "What are the B.Tech fees and what are the hostel facilities?"
    Sub-queries: What are the B.Tech fees?, What are the hostel facilities?

    Query: "Compare the CSE and EE departments"
    Sub-queries: Compare the CSE and EE departments?

    Query: "How is placement of CSE and EE students?"
    Sub-queries: How is placement of CSE students?, How is placement of EE students?

    Query: "What are the B.Tech fees and what are the hostel facilities?"
    Sub-queries: What are the B.Tech fees?, What are the hostel facilities?
    ----
    
    Query: {query}
    Sub-queries:
    """

    def __init__(self, llm: llms.BaseChat, **kwargs):
        super().__init__(
            llm=llm,
            prompt_template=self.PROMPT_TEMPLATE,
            **kwargs,
        )





# chatmodel  = llms.LiteLLMChat(
#     model="groq/llama3-8b-8192",
# )

# bot = SubQueryAgent(
#     llm=chatmodel
# )

# server = CustomServer(
#     host="0.0.0.0",
#     port=8000,
#     router_agent_answerer=bot,
# )

# server.run()
