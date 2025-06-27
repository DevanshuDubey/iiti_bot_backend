import BaseAgent
from pathway.xpacks.llm import llms, servers
import pathway as pw
import json
x = json.load()

class CritiqueAgent(BaseAgent.BaseAgent):
    PROMPT_TEMPLATE = """You are an expert Relevance and Quality Evaluation Agent. Your task is to **critically evaluate** the given answer with respect to the user query and the provided context (retrieved documents). Your evaluation must consider multiple dimensions to determine a **Relevancy Score** between 0 and 1 and provide **detailed constructive feedback** for improvement.

### Evaluation Criteria:

1. **Correctness**: Is the information factually accurate and aligned with the context?
2. **Completeness**: Does the answer cover all key elements in the context that are relevant to the query?
3. **Clarity**: Is the answer clear and unambiguous from the user's perspective?
4. **Faithfulness to Query and Context**: 
   - Is the context relevant and sufficient for answering the query?
   - Does the answer rely only on the context without hallucinating?
5. **User Satisfaction**: Will the user likely feel the answer is informative, useful, and complete?

### Scoring Guide:
- A score **above 0.80** should only be assigned if the answer is factually correct, complete, and does not need further refinement.
- A score **below or equal to 0.80** indicates that the answer is either partially correct, incomplete, unclear, or not aligned with the query/context.

### Output Format:
Return a **valid JSON object** with:
- `"SCORE"`: A float between 0.0 and 1.0 (rounded to 2 decimal places)
- `"FEEDBACK"`: 
   - If SCORE > 0.80: `"PASS"`
   - If SCORE ≤ 0.80: **Provide highly detailed, constructive feedback**, highlighting missing information, factual errors, lack of clarity, or irrelevance. Feedback will be used to improve the next generated answer.

    
    
    ---
    [EXAMPLES]
    Query: "where is iit indore"
    Context: "IIT Indore is a 2nd Generation IIT located in simrol village of Indore, MP"
    Answer: "IIT Indore is located in simrol village of Indore, MP"
    ```json
    {
        "SCORE": 0.95,
        "FEEDBACK": "PASS"
    }

    Query: "where is iit indore"
    Context: "IIT Indore is a 2nd Generation IIT located in simrol village of Indore, MP"
    Answer: "IIT Indore is located in Indore, MP"
    ```json
    {
        "SCORE": 0.60,
        "FEEDBACK": "The answer dosen't contain all relevant information available in the context. It is missing which village is the iit indore situated in"
    }


    Query: "what courses are offered at iit indore"
    Context: "IIT Indore is a 2nd Generation IIT located in Indore, MP. It offers a total of 9 B.Tech courses (CSE, MnC, EE, ME, EP, CE, MEMS, CE, SSE), some M.Tech course, some PhD courses and B.Design courses"
    Answer: "IIT Indore offers 9 B.Tech courses"
    ```json
    {
        "SCORE": 0.60,
        "FEEDBACK": "Although answer provided is correct, but it is not complete. Details about which B.Tech courses are offered and about other courses offered like M.Tech, PhD and B.Design are missing" 
    }

    Query: "What are the B.Tech fees and what are the hostel facilities?"
    Context: "IIT Indore is a 2nd Generation IIT located in Indore, MP. It offers a total of 9 B.Tech courses (CSE, MnC, EE, ME, EP, CE, MEMS, CE, SSE), some M.Tech course, some PhD courses and B.Design courses. Fee structure of all these courses range from 2.5lakhs to 3.5lakhs per year."
    Answer: "B Tech fees ranges is 1.5lakhs"
    ```json
    {
        "SCORE": 0.05,
        "FEEDBACK": "Given answer is incorrect, the fees ranges from 2.5 to 3.5 lakhs per year." 
    }
    
    
    [USER QUERY]
    User Query: "{query}"
    Context: "{context}"
    Answer: "{answer}"
    """

    def __init__(self, llm: llms.BaseChat, **kwargs):
        super().__init__(
            llm=llm,
            prompt_template=self.PROMPT_TEMPLATE,
            **kwargs,
        )

    @pw.table_transformer 
    def answer_query(self, queries: pw.Table) -> pw.Table:
    
        prompt = self.prompt_template.format(query=queries.query,
                                             context="/n/n".join(queries.docs[:k]),
                                             answer=queries.answer)
        response= self.llm(llms.prompt_chat_singe_qa(prompt), model=self.default_llm_name)
        response= json.load(response)
        
        
        queries = queries.with_columns(
            score=response["SCORE"],
            feedback=response["FEEDBACK"]
        )
        
        return queries
    
