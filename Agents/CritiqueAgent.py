import json
from typing import Optional, Dict, Any
from pathway.xpacks.llm.llms import LiteLLMChat

class CritiqueAgent:
    def __init__(self, model: str = "groq/llama3-70b-8192", prompt_template: Optional[str] = None):
        
        self.chat = LiteLLMChat(
            model=model,
            response_format={"type": "json_object"},
        )
        self.prompt_template = prompt_template or self._default_template()

    def _default_template(self) -> str:
        # The prompt is already well-defined for returning JSON. No changes needed.
        return """You are an expert Relevance and Quality Evaluation Agent. Your task is to *critically evaluate* the given answer with respect to the user query and the provided context (retrieved documents). Your evaluation must consider multiple dimensions to determine a *Relevancy Score* between 0 and 1 and provide *detailed constructive feedback* for improvement.

Evaluation Criteria:

1.  **Correctness:** Is the information factually accurate and perfectly aligned with the `context`? Are there any subtle inaccuracies or misrepresentations?
2.  **Completeness & Depth:** Does the answer extract *all* relevant facts, figures, and key details from the `context` that are necessary to fully satisfy the `user query`? Does it go beyond surface-level information if the context supports it?
3.  **Conciseness:** Is the answer direct and to the point? Does it avoid redundant phrasing or unnecessary information from the context that doesn't directly address the query?
4.  **Faithfulness:**
    - **To Context:** Does the answer *only* use information from the provided `context`? Any information not present in the context (a hallucination) is a critical failure.
    - **To Query:** Does the answer directly address the user's specific question, or does it provide related but irrelevant information?
5.  **Clarity & Formatting:** Is the answer easy to understand, well-structured, and unambiguous? (e.g., using lists for multiple items, clear sentences).

Assign a `SCORE` from 0.0 to 1.0. Be strict. A high score is a reward for excellence.

-   **0.9 - 1.0 (Excellent / PASS):** The answer is perfect or near-perfect. It is correct, complete, concise, and faithful. It fully satisfies the user's intent and requires no refinement. A score of 1.0 should be reserved for flawless answers.
-   **0.7 - 0.89 (Good but Flawed / FAIL):** The answer is good and addresses the main points but has minor flaws. It might be missing a non-critical detail, have slightly awkward phrasing, or be a bit verbose.
-   **0.5 - 0.69 (Partially Correct / FAIL):** The answer is partially correct but has significant omissions. It addresses the query at a surface level but misses crucial information present in the context.
-   **0.2 - 0.49 (Mostly Incorrect or Irrelevant / FAIL):** The answer contains significant factual errors, is largely incomplete, or provides information that is mostly irrelevant to the query.
-   **0.0 - 0.19 (Complete Failure / FAIL):** The answer is entirely incorrect, contains hallucinations (information not from the context), or completely fails to address the query, or the logic of the answer is wrong (eg A implies B, B implies C, D implies C,  but answer says A implies D. This is wrong logic in the answer )

Feedback Guide:
- If `SCORE` is above 0.89 then feedback is "PASS"
- Give Feedback properly about why the given answer does not PASS, explaining what this from the main query are left out or not properly considered in the answer.
- The Feedback given will be used in the answer generating agent to generate back a good answer hence provide the feedback in such a way that it is useful to the answer generating agent if provided with same context and query (Without the old answer). 
- Also provide feedback when something related to query is present in docs but not included in the answer.

Output Format:

Return ONLY a valid JSON object with two keys:
- "SCORE": A float between 0.0 and 1.0 (rounded to 2 decimal places).
- "FEEDBACK":
    - If SCORE > 0.80: "PASS"
    - If SCORE ≤ 0.80: Provide highly detailed, constructive feedback, highlighting missing information, factual errors, lack of clarity, or irrelevance. This feedback will be used to improve the next generated answer.

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
{
    "SCORE": 0.60,
    "FEEDBACK": "The answer should also contain the information iit indore is located in simrol."
}
Query: "what courses are offered at iit indore"
Context: "IIT Indore is a 2nd Generation IIT located in Indore, MP. It offers a total of 9 B.Tech courses (CSE, MnC, EE, ME, EP, CE, MEMS, CE, SSE), some M.Tech course, some PhD courses and B.Design courses"
Answer: "IIT Indore offers 9 B.Tech courses"
{
    "SCORE": 0.30,
    "FEEDBACK": "The answer should list all the available courses rather than just saying the total count. It should also mention about M.Tech, PhD and B.Design courses."
}
Query: "Compare CSE and EE department"
Context: "IIT Indore is a 2nd Generation IIT located in Indore, MP. CSE in IIT Indore is the best Branch in IIT Indore, with average placement of 47lpa. It contains multiples courses like DSA, DBIS, DMS, AI/ML etc. It has an average branch strength of nearly 83 every year. The EE branch is also a good branch in IIT Indore with average package of 24lpa, offering courses like Basic Electronics, Complex Electronics etc. Its average Branch strength is 85 every year"
Answer: "Both CSE and EE are good branches in IIT Indore with CSE having a good edge over EE in terms of placment, which is nearly double of EE. The average placement of CSE is 47lpa, roughly double of EE's 24lpa. Their courses also differ, CSE offering courses like DSA, DBIS, DMS, AI/ML whereas EE offering Basic Electronics and Complex Electronics"
{
    "SCORE": 0.95
    "FEEDBACK": "PASS"
}
"""

    def set_prompt_template(self, new_template: str):
        self.prompt_template = new_template
    
    def run(self, query: str, docs: str, answer: str) -> Dict[str, Any]:
        
        user_prompt = f'User Query: "{query}"\nContext: "{docs}"\nAnswer: "{answer}"'
    
        messages = [
            {"role": "system", "content": self.prompt_template},
            {"role": "user", "content": user_prompt},
        ]
    
        response_text = self.chat.__wrapped__(messages=messages)
        return self._parse_json_response(response_text)
    
    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """
        Safely parse the LLM's JSON response.
        """
    
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            return {
                "error": f"Failed to decode JSON from response: {str(e)}",
                "raw_response": text
            }
    
    
 