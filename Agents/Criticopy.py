import asyncio
import json
from typing import Optional, Dict, Any, List

# Make sure you have pathway and the xpack-llm extras installed
# pip install "pathway[xpacks-llm]"
from pathway.xpacks.llm.llms import LiteLLMChat

class CritiqueAgent:
    """
    An agent that uses a LiteLLM-compatible model to evaluate an answer
    based on a query and context.
    """
    def __init__(self, model: str = "groq/llama3-70b-8192", prompt_template: Optional[str] = None):
        """
        Initializes the CritiqueAgent.

        Args:
            model (str): The model identifier for LiteLLM (e.g., "groq/llama3-70b-8192").
            prompt_template (Optional[str]): An optional custom prompt template.
        """
        # Instantiate the LiteLLMChat UDF from Pathway.
        # We configure it to return JSON for easier parsing.
        self.chat = LiteLLMChat(
            model=model,
            # Instruct the model to return a valid JSON object.
            # This is supported by many models, including Groq's Llama3.
            response_format={"type": "json_object"},
            # You can also set other litellm parameters here, like temperature
            # temperature=0.1,
        )
        self.prompt_template = prompt_template or self._default_template()

    def _default_template(self) -> str:
        # The prompt is already well-defined for returning JSON. No changes needed.
        return """You are an expert Relevance and Quality Evaluation Agent. Your task is to **critically evaluate** the given answer with respect to the user query and the provided context (retrieved documents). Your evaluation must consider multiple dimensions to determine a **Relevancy Score** between 0 and 1 and provide **detailed constructive feedback** for improvement.

Evaluation Criteria:

Correctness: Is the information factually accurate and aligned with the context?
Completeness: Does the answer cover all key elements in the context that are relevant to the query?
Clarity: Is the answer clear and unambiguous from the user's perspective?
Faithfulness to Query and Context:
- Is the context relevant and sufficient for answering the query?
- Does the answer rely only on the context without hallucinating?
User Satisfaction: Will the user likely feel the answer is informative, useful, and complete?

Scoring Guide:

- A score above 0.80 should only be assigned if the answer is factually correct, complete, and does not need further refinement.
- A score below or equal to 0.80 indicates that the answer is either partially correct, incomplete, unclear, or not aligned with the query/context.

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
    "FEEDBACK": "The answer dosen't contain all relevant information available in the context. It is missing which village is the iit indore situated in"
}
Query: "what courses are offered at iit indore"
Context: "IIT Indore is a 2nd Generation IIT located in Indore, MP. It offers a total of 9 B.Tech courses (CSE, MnC, EE, ME, EP, CE, MEMS, CE, SSE), some M.Tech course, some PhD courses and B.Design courses"
Answer: "IIT Indore offers 9 B.Tech courses"
{
    "SCORE": 0.60,
    "FEEDBACK": "Although answer provided is correct, but it is not complete. Details about which B.Tech courses are offered and about other courses offered like M.Tech, PhD and B.Design are missing"
}
"""
    

    def set_prompt_template(self, new_template: str):
        self.prompt_template = new_template
    
    def run(self, query: str, docs: str, answer: str) -> Dict[str, Any]:
        """
        Asynchronously runs the evaluation.
    
        Args:
            query (str): The user's query.
            docs (str): The context documents.
            answer (str): The answer to be evaluated.
    
        Returns:
            A dictionary containing the SCORE and FEEDBACK.
        """
        user_prompt = f'User Query: "{query}"\nContext: "{docs}"\nAnswer: "{answer}"'
    
        messages = [
            {"role": "system", "content": self.prompt_template},
            {"role": "user", "content": user_prompt},
        ]
    
        # Call the __wrapped__ async method of the LiteLLMChat instance
        response_text = self.chat.__wrapped__(messages=messages)
    
        return self._parse_json_response(response_text)
    
    def _parse_json_response(self, text: str) -> Dict[str, Any]:
        """
        Safely parse the LLM's JSON response.
        """
        if not text:
            return {"error": "Received an empty response from the model."}
        try:
            # The model was instructed to return JSON, so we can parse it directly.
            return json.loads(text)
        except json.JSONDecodeError as e:
            return {
                "error": f"Failed to decode JSON from response: {str(e)}",
                "raw_response": text
            }
    
    
 