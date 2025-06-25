from BaseAgent import BaseAgent, CustomServer
from pathway.xpacks.llm import llms
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)


class ClarifyingAgent(BaseAgent):
    PROMPT_TEMPLATE = """You are an expert Clarifying Agent who asks back questions to clarify the user query.

    ---
    
    [USER QUERY]
    User Query: "{query}"
    Clarification_queestion:"""

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
#     port=8080,
#     router_agent_answerer=bot
# )

# server.run()