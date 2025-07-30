from .BaseAgent import BaseAgent 
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


