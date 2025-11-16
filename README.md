# IITI-Bot

**IITI-Bot** is an intelligent chatbot designed to serve as an accurate information source for the Indian Institute of Technology Indore (IITI). It leverages Pathway's dynamic dataflow and dynamically reasons across documents to deliver trustworthy and context-rich responses. The core of our chatbot is a multi-agent Retrieval-Augmented Generation (RAG) system, which allows for specialized agents to collaborate on providing the most accurate and relevant information.

From academic programs and faculty information to campus facilities and events, IITI-BOT provides precise information.

## Project Presentation

For a more detailed overview and demonstration of the project, you can view our presentation on Canva:

**[View the Presentation](https://www.canva.com/design/DAGvkW1ppnQ/V1Xw_DrFUdDmb6lNO8MJfw/view)**

## Deployed at

You can interact with IITI-Bot at: **https://iit-indore-bot.vercel.app/**

### A Note on "Cold Starts"

When the website receives a query after a long period of inactivity, it may take 10-15 minutes to respond. This is due to a "cold restart" of the service, which involves re-indexing the necessary documents. After this initial query, all subsequent queries will be answered instantly. This is a common occurrence in serverless environments where resources are allocated on demand.

## How it Works

IITI-Bot is built upon a sophisticated multi-agent Retrieval-Augmented Generation (RAG) system. This means that instead of a single AI model handling everything, we have a team of specialized AI agents working together. Here's a simplified breakdown of the process:

1.  **Query Understanding:** When you ask a question, a "manager" agent first analyzes your query to understand its intent.
2.  **Specialized Retrieval:** The manager then routes the query to the most appropriate "retriever" agent. We have different retriever agents, each an expert in a specific domain of information related to IIT Indore (e.g., one for academic programs, another for campus facilities).
3.  **Information Gathering:** The selected retriever agent then searches through its dedicated knowledge base to find the most relevant documents and information to answer your question.
4.  **Response Generation:** The retrieved information is then passed to a "generator" agent. This agent synthesizes the information into a coherent and easy-to-understand response.
5.  **Trustworthy and Context-Rich Answers:** By dynamically reasoning across the retrieved documents, IITI-Bot can provide you with accurate and contextually relevant answers.

This multi-agent approach, powered by Pathway's dynamic dataflow, allows for a more efficient and accurate retrieval process, leading to higher quality responses.

## Features

*   **Accurate Information:** Get precise and reliable information about various aspects of IIT Indore.
*   **Context-Rich Responses:** The chatbot understands the context of your queries to provide more relevant answers.
*   **Multi-Agent RAG System:** A team of specialized AI agents collaborates to deliver the best possible response.
*   **Powered by Pathway:** Utilizes Pathway's dynamic dataflow for efficient and real-time data processing.
*   **Comprehensive Knowledge Base:** Covers a wide range of topics including:
    *   Academic Programs
    *   Admission Procedures
    *   Faculty and Research
    *   Campus Facilities
    *   Student Life and Events
    *   And much more!

We are constantly working to improve IITI-Bot and expand its knowledge base. We welcome your feedback and suggestions!
