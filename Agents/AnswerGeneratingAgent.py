import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

from dotenv import load_dotenv
load_dotenv()

import pathway as pw
from pathway.xpacks.llm import llms, servers
from pathway.xpacks.llm.question_answering import BaseQuestionAnswerer

@pw.udf
def create_result_json(query: str, response: str) -> pw.Json:
    return pw.Json({"query": query, "response": response})

class AnswerGeneratingAgent(BaseQuestionAnswerer):
    prompt_template="""You are an expert Answer Generating Agent. Your primary mission is to provide accurate, well-structured, and logical answers based *only* on the provided documents (`docs`). You must adhere to the following instructions strictly.

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

    def __init__(
        self,
        llm: llms.BaseChat,
        *,
        default_llm_name: str | None = None,
        prompt_template: str | None = None,
    ):
        self.llm = llm
        if default_llm_name is None:
            default_llm_name = llm.model

        if prompt_template is None:
            self.prompt_template = self.__class__.prompt_template
        else:
            self.prompt_template = prompt_template
        
        self._init_schemas(default_llm_name)

    def _init_schemas(self, default_llm_name: str | None = None) -> None:
        class BaseQuerySchema(pw.Schema):
            query: str
            model: str | None = pw.column_definition(default_value=default_llm_name)
            docs: list[str] | None = pw.column_definition(default_value=[])
            k: int = pw.column_definition(default_value=3)
        self.AnswerQuerySchema = BaseQuerySchema
        self.RetrieveQuerySchema = pw.Schema
        self.StatisticsQuerySchema = pw.Schema
        self.InputsQuerySchema = pw.Schema

    @pw.table_transformer
    def answer_query_table(self, queries: pw.Table) -> pw.Table:
        """
        Returns answer as pw.Table
        """
        results = queries.with_columns(
            prompt=pw.apply(
                lambda query, docs, k: self.prompt_template.format(
                    query=query, docs="\n\n".join(docs[:k])
                ),
                pw.this.query, pw.this.docs, pw.this.k
            )
        )

        results = results.with_columns(
            response=self.llm(llms.prompt_chat_single_qa(pw.this.prompt), model=pw.this.model)
        )
        return results

    @pw.table_transformer
    def answer_query(self, queries: pw.Table) -> pw.Table:
        """
        Returns answer as Json
        """
        return self.answer_query_table(queries).select(
            result=create_result_json(pw.this.query, pw.this.response)
        )

# class CustomServer(servers.BaseRestServer):
#     def __init__(
#         self,
#         host: str,
#         port: int,
#         answerer: "AnswerGeneratingAgent",
#         **rest_kwargs,
#     ):
#         super().__init__(host, port, **rest_kwargs)
#         self.serve(
#             route="/v1/chat",
#             schema=answerer.AnswerQuerySchema,
#             handler=answerer.answer_query,
#             **rest_kwargs,
#         )

# chatmodel = llms.LiteLLMChat(
#     model="groq/llama3-8b-8192",
# )

# bot = AnswerGeneratingAgent(
#     llm=chatmodel
# )

# server = CustomServer(
#     host="0.0.0.0",
#     port=8000,
#     answerer=bot,
# )

# server.run()