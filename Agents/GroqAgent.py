import re
import ast
from groq import Groq
from typing import Optional, Dict, List, Any

class GroqAgent:
    def __init__(self, model: str = "llama3-70b-8192", prompt_template: Optional[str] = None):
        self.client = Groq()
        self.model = model
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
❌ Never use prior knowledge.
❌ Never hallucinate.
❌ Never answer if not grounded in context.
✅ All fields in the output dictionary are mandatory.
✅ Deduplicate overlapping or repeated snippets before output.
"""

    def set_prompt_template(self, new_template: str):
        self.prompt_template = new_template

    def run(self, query: str, docs: str, feedback: Optional[str] = "") -> Dict[str, Any]:
        user_prompt = f"""Query:
{query}

Docs:
{docs}

Previous Feedback (if any):
{feedback}"""
        
        response = self.client.chat.completions.create(
      model=self.model,
      messages=[
          {"role": "system", "content": self._default_template()},
          {"role": "user", "content": user_prompt}
      ]
  )

        raw_text = response.choices[0].message.content.strip()
        result_dict = self._extract_dict_from_response(raw_text)

        return result_dict
    def _extract_dict_from_response(self, text: str) -> Dict[str, Any]:
        """
Safely parse the LLM response assuming it contains a Python-style or JSON-style dictionary.
"""
        try:
# Extract the first JSON/dict-like block
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