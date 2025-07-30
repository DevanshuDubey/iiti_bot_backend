from .BaseAgent import BaseAgent
from pathway.xpacks.llm import llms
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)


class RouterAgent(BaseAgent):
    PROMPT_TEMPLATE = """You are an expert query router. Your task is to classify a user's query into ONE of the following categories based on the structure of the question itself. Respond with ONLY the category name.

    ---
    [CATEGORIES AND DEFINITIONS]
    1. clarifying_agent: The query is too vague or ambiguous to be answered directly or requires clarification from the user.
    2. chat_agent: The query is a greeting, a sign-off, or clearly off-topic from academics or IIT Indore.
    3. sub_query_generating_agent: The query is simple single-toic or EXPLICITLY asks MULTIPLE questions (e.g., using "and"), asks for a COMPARISON (e.g., "compare", "vs"), or asks for a relationship between two distinct topics.
    
    ---
    [EXAMPLES]
    Query: "where is iit indore"
    Category: sub_query_generating_agent
    
    Query: "what courses are offered at iit indore"
    Category: sub_query_generating_agent
    
    Query: "hi there!"
    Category: chat_agent
    
    Query: "What are the B.Tech fees and what are the hostel facilities?"
    Category: sub_query_generating_agent
    
    Query: "Compare the CSE and EE departments"
    Category: sub_query_generating_agent
    
    Query: "is it true that"
    Category: clarifying_agent
    ---
    
    [USER QUERY]
    User Query: "{query}"
    Category:"""

    def __init__(self, llm: llms.BaseChat, **kwargs):
        super().__init__(
            llm=llm,
            prompt_template=self.PROMPT_TEMPLATE,
            **kwargs,
        )
 


