import re
import ast
from typing import Optional, Dict, Any
from pathway.xpacks.llm.llms import LiteLLMChat

class AnswerGeneratingAgent:
    def __init__(self, model: str = "groq/llama3-70b-8192", prompt_template: Optional[str] = None):
        self.chat = LiteLLMChat(
            model=model,
            response_format={"type": "json_object"},
        )
        self.prompt_template = prompt_template or self._default_template()

    def _default_template(self) -> str:
        return """You are a highly capable Answer Generation Agent. Your task is to synthesize a high-quality answer to a user query using only the provided context.

### TASK INSTRUCTIONS:

1. Read the provided query and its corresponding context(s) carefully.
2. If the query is a composite or includes multiple sub-questions, you must answer them all in a logically unified, non-repetitive way.
3. Your answer must be:
   - Coherent
   - Well-structured
   - Free from duplication
   - Grounded **strictly** in the provided context(s).
4. At the end, extract the **minimal set of unique, supporting evidence snippets** from the docs that justify your answer.
5. Return the output as a **valid Python dictionary** using this format:

```json
{
  "answer": "<Final synthesized answer here>",
  "source_snippet": "<Summarised text source snippets from where the answer is taken>"
}
CONSTRAINTS:
*** Never use prior knowledge.
*** Never hallucinate.
*** Never answer if not grounded in context.
*** All fields in the output dictionary are mandatory.
*** Deduplicate overlapping or repeated snippets before output.
"""

    def set_prompt_template(self, new_template: str):
        self.prompt_template = new_template

    def run(self, query: str, docs: str, feedback: Optional[str] = "") -> Dict[str, Any]:
        user_prompt = f"""Query: "{query}"\nDocs:"{docs}"\nPrevious Feedback (if any):"{feedback}" \n"""
        messages = [
            {"role": "system", "content": self.prompt_template},
            {"role": "user", "content": user_prompt},
        ]

        response_text = self.chat.__wrapped__(messages=messages)

        return self._extract_dict_from_response(response_text)
    
    
    def _extract_dict_from_response(self, text: str) -> Dict[str, Any]:
        try:
            match = re.search(r"{.*}", text, re.DOTALL)
            if match:
                extracted = match.group(0)
                parsed = ast.literal_eval(extracted) # safer than eval
                return parsed
            else:
                raise ValueError("No valid dict structure found.")
        except Exception as e:
            return {
"error": f"Failed to parse response: {str(e)}",
"raw_response": text
}