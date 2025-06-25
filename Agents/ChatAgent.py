from BaseAgent import BaseAgent, CustomServer
from pathway.xpacks.llm import llms
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)


class ChatAgent(BaseAgent):
    PROMPT_TEMPLATE = """You are a general chat agent named "IITI_BOT" made by devanshu_dubey to chat with user regarding IIT INDORE.
    User Query: "{query}"
    Your response:"""

    def __init__(self, llm: llms.BaseChat, **kwargs):
        super().__init__(
            llm=llm,
            prompt_template=self.PROMPT_TEMPLATE,
            **kwargs,
        )
 






# chatmodel  = llms.LiteLLMChat(
#     model="groq/llama3-8b-8192",
# )

# bot = ChatAgent(
#     llm=chatmodel
# )

# server = CustomServer(
#     host="0.0.0.0",
#     port=8080,
#     router_agent_answerer=bot,
# )

# server.run()