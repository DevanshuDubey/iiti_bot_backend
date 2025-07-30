from BaseAgent import BaseAgent, CustomServer
from pathway.xpacks.llm import llms
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)


class ClarifyingAgent(BaseAgent):
    PROMPT_TEMPLATE = """You are a Clarifying Agent designed to help resolve ambiguity in user queries.

Your task is to ask a clear, concise, and helpful follow-up question that will guide the user to provide more specific information, so their query can be better understood and routed to the appropriate expert.

Do not answer the query. Only ask **one** clarifying question based on the information provided.

---

[USER QUERY]
"{query}"

Clarifying Question:"""


    def __init__(self, llm: llms.BaseChat, **kwargs):
        super().__init__(
            llm=llm,
            prompt_template=self.PROMPT_TEMPLATE,
            **kwargs,
        )



# chatmodel  = llms.LiteLLMChat(
#     model="groq/llama3-8b-8192",
# )

# bot = ClarifyingAgent(
#     llm=chatmodel
# )

# server = CustomServer(
#     host="0.0.0.0",
#     port=8000,
#     router_agent_answerer=bot
# )

# server.run()
