from BaseAgent import BaseAgent, CustomServer
from pathway.xpacks.llm import llms
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)


class TopicKeywordsAgent(BaseAgent):
    PROMPT_TEMPLATE = """You are an expert query router. Your task is to classify a user's query into ONE of the following categories based on the structure of the question itself. Respond with ONLY the category names separated by commas.
    [CATEGORIES AND DEFINITIONS]
    0. general: The query is a general question regarding IIT Indore.
    1. hostel: The query asks about hostel facilities.
    2. fees: The query asks about fees.
    3. courses_btech: The query asks about B.Tech courses.
    4. courses_mtech: The query asks about M.Tech courses.
    5. courses_phd: The query asks about Ph.D. courses.
    6. courses_general: The query asks about general courses.
    7. admission: The query asks about admission.
    8. placement: The query asks about placement.
    9. faculty: The query asks about faculty.
    10. research: The query asks about research.
    11. alumni: The query asks about alumni.
    12. events: The query asks about events.
    13. scholarships: The query asks about scholarships.
    14. student_life: The query asks about student life.
    15. student_support: The query asks about student support.
    16. research_facilities: The query asks about research facilities.
    17. library: The query asks about library.
    18. canteen: The query asks about canteen.
    19. sports: The query asks about sports.
    20. cultural_events: The query asks about cultural events.
    21. departments: The query asks about departments.
    22. other: The query asks about anything else.

    ---
    [EXAMPLES]
    Query: "where is iit indore"
    Category: other

    Query: "what courses are offered at iit indore"
    Category: courses_general

    Query: "hi there!"
    Category: other

    Query: "What are the B.Tech fees and what are the hostel facilities?"
    Category: fees, hostel

    Query: "Compare the CSE and EE departments"
    Category: departments

    [QUERY]
    Query: {query}
    Respond with ONLY the category names separated by commas.
    Nothing else.
    Category:
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

# bot = TopicKeywordsAgent(
#     llm=chatmodel
# )

# server = CustomServer(
#     host="0.0.0.0",
#     port=8000,
#     router_agent_answerer=bot,
# )

# server.run()