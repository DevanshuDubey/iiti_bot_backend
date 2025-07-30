from BaseAgent import BaseAgent, CustomServer
from pathway.xpacks.llm import llms
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

class SubQueryAgent(BaseAgent):
    PROMPT_TEMPLATE = """
You are an expert Sub-Query Generator. Your task is to break down a user query into smaller sub-queries **if it contains multiple parts, comparisons, or discusses more than one topic**.

In all cases, **enhance each sub-query** to make it clearer, more complete, or more formal.  
If the original query is simple and self-contained, return it as a single enhanced sub-query without splitting.

Respond with ONLY the enhanced sub-queries, separated by <SBQ>.

----
[EXAMPLES - COMPLEX QUERIES]
Query: "What are the B.Tech fees and what are the hostel facilities?"
Sub-queries: What is the fee structure for the B.Tech program at IIT Indore? <SBQ> What hostel facilities are available for students at IIT Indore?

Query: "Compare the CSE and EE departments"
Sub-queries: What are the key features of the CSE department at IIT Indore? <SBQ> What are the key features of the EE department at IIT Indore?

Query: "How is placement of CSE and EE students?"
Sub-queries: How are the placement opportunities for CSE students at IIT Indore? <SBQ> How are the placement opportunities for EE students at IIT Indore?

Query: "What is the cutoff and fee structure for M.Tech?"
Sub-queries: What is the admission cutoff for the M.Tech program at IIT Indore? <SBQ> What is the fee structure for the M.Tech program at IIT Indore?

----
[EXAMPLES - SIMPLE QUERIES]
Query: "What are the B.Tech fees?"
Sub-queries: What is the fee structure for the B.Tech program at IIT Indore?

Query: "How is campus life at IIT Indore?"
Sub-queries: What is student life like on the IIT Indore campus?

Query: "Does IIT Indore have a gym?"
Sub-queries: Are gym facilities available to students at IIT Indore?

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
