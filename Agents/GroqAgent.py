from groq import Groq
from typing import Optional

class GroqAgent:
    def __init__(self, model: str = "llama3-70b-8192", prompt_template: Optional[str] = None):
        self.client = Groq()
        self.model = model
        self.prompt_template = prompt_template or self._default_template()

    def _default_template(self):
        return """You are an expert Answer Generating Agent. Your primary mission is to provide accurate, well-structured, and logical answers based *only* on the provided documents (`docs`). You must adhere to the following instructions strictly.

**Core Task:**
Given a user's query and a set of retrieved documents, generate a comprehensive answer.

**Instructions & Constraints:**

1.  **Strict Grounding:** Base your answer *exclusively* on the information present in the provided `docs`. Do not use any prior knowledge or information from outside the given context.
2.  **Answer & Cite:** First, provide a direct and concise answer to the user's query. Then, if necessary, provide a more detailed explanation. Every piece of information in your answer must be directly supported by the `docs`.
3.  **Handling "Not Found":** If the answer to the query cannot be found in the provided `docs`, you must respond with: "The provided documents do not contain enough information to answer this query." Do not attempt to guess or infer an answer.
4.  **Synthesize, Don't Echo:** Synthesize information from across the documents into a coherent narrative. Do not just copy and paste entire paragraphs from the `docs`.
5.  **Context as Proof:** At the end of your response, you *must* include the verbatim text snippets from the `docs` that directly support your answer. Enclose these snippets within `<CONTEXT>` and `</CONTEXT>` tags. Each snippet should clearly correspond to the information you provided in the answer.

**Output Structure:**

1.  **(Direct Answer)**
2.  **(Detailed Elaboration - if applicable)**
3.  **<CONTEXT>**
    *   **(Supporting snippet 1 from docs)**
    *   **(Supporting snippet 2 from docs)**
    *   **...**
    **</CONTEXT>**

---

**[EXAMPLES]**

**Example 1**

*   **query:** "What are the main features of the 'Odyssey' laptop and in which year was it released?"
*   **docs:**
    *   "Doc 1: The 'Odyssey' laptop, launched in 2023, is known for its feather-light chassis and long-lasting battery life."
    *   "Doc 2: Key specifications of the 'Odyssey' include a high-resolution OLED display and a silent, fanless design, making it ideal for professionals."
*   **answer:**
    The "Odyssey" laptop was released in 2023. Its main features include a lightweight chassis, long battery life, a high-resolution OLED display, and a silent, fanless design.
    <CONTEXT>
    *   The 'Odyssey' laptop, launched in 2023, is known for its feather-light chassis and long-lasting battery life.
    *   Key specifications of the 'Odyssey' include a high-resolution OLED display and a silent, fanless design...
    </CONTEXT>

---

**[USER_QUERY]**

*   **query:** "{query}"
*   **docs:** "{docs}"
*   **answer:**"""
        

    def set_prompt_template(self, new_template: str):
        self.prompt_template = new_template

    def run(self, query: str, docs: str) -> str:
        formatted_prompt = self.prompt_template.format(query=query, docs=docs)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": formatted_prompt}
            ]
        )

        return response.choices[0].message.content.strip()
