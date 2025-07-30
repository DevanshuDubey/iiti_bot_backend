import requests
from typing import Tuple
from Agents.answerGeneratingAgent import AnswerGeneratingAgent
from Agents.critiqueAgent import CritiqueAgent
 
document_store_url = "http://0.0.0.0:8000/v1/retrieve"
 
def get_payload(query: str, k:int) -> dict:
    """
    Formatting the payload to send request to the document store
    """
    payload = {
    "query": query,
    "k": k,
    "metadata_filter": None,
    "filepath_globpattern": None
}
    return payload


def single_query(main_query:str, k : int) -> Tuple[str, str]:
    """
    Retrieving and formatting documents for a single query
    """
    query_string = f"Main query : {main_query}\n"
    doc_string = ""
    payload = get_payload(main_query, k)
    response = requests.post(document_store_url, json=payload)
    results = response.json()
    for doc in results:
        doc_string += doc["text"]

    return query_string , doc_string    


def multiple_queries(main_query:str, queries : list, k : int) -> Tuple[str, str]:
    """
    Retrieving and formatting documents for multiple subqueries
    """
    query_string = f"Initial main query : {main_query}\n"
    doc_string = ""
    for i , query in enumerate(queries):
        query_string += f"Query{i+1} : {query}\n"
        doc_string += f"\n\nContext for query{i+1} :\n"
        payload = get_payload(query, k)
        response = requests.post(document_store_url, json=payload)
        results = response.json()
        for doc in results:
            doc_string += doc["text"]

    return query_string , doc_string        

           

def sub_pipeline(main_query:str, queries : list, k : int) :
    iteration_counter = 0
    max_adaptive_iterations = 3
    initial_k = k
    critique_threshold = 0.8
    feedback = "There is no feedback as of now as this your first try to answer this question."
    subqueries = True if len(queries) > 1 else False
    if subqueries:
        query_string, doc_string = multiple_queries(main_query, queries, initial_k)
    else:
        query_string, doc_string = single_query(main_query, initial_k)
    
    AGAagent = AnswerGeneratingAgent()
    critiqueAgent = CritiqueAgent()
    
    while True:
        response = AGAagent.run(query_string, doc_string, feedback)
        answer = response["answer"]
        source_snippet = response["source_snippet"]

        critique_response = critiqueAgent.run(main_query, doc_string, answer)
        answer_score = critique_response["SCORE"]
        feedback = critique_response["FEEDBACK"]
         
        if answer_score < critique_threshold:

            if iteration_counter == max_adaptive_iterations:
               fallback_response_text = """
                I'm sorry, I couldn't find an exact answer to your query in the provided context. 
                However, I've included the most relevant source snippets below that may help you find the information you're looking for.
                """
               fallback_response = {
                   "text" : fallback_response_text,
                   "source_snippet" : source_snippet
               }
               return fallback_response , answer_score , feedback , iteration_counter , doc_string , query_string
            
            iteration_counter += 1 
            if subqueries:
                query_string, doc_string = multiple_queries(main_query, queries, initial_k + iteration_counter*2)
            else:
                query_string, doc_string = single_query(main_query, initial_k + iteration_counter*2)
      
        else:
            final_response = {
                   "text" : answer,
                   "source_snippet" : source_snippet
               }
            return final_response , answer_score , feedback , iteration_counter , doc_string , query_string



         











     
     
     
         
         
         