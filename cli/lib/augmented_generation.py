import json

from .hybrid_search import HybridSearch
from .search_utils import load_movies
from .gemini_utils import generate_response


def augmented_generation(query):
    hs = HybridSearch(load_movies())
    relevant_documents = hs.rrf_search(query, k=60, limit=5)
    response = generate_response(
        f"""
        Answer the question or provide information based on the provided documents. This should be tailored to SilasStreaming users. SilasStreaming is a movie streaming service.

        Query: {query}

        Documents:
        {json.dumps(relevant_documents)}

        Provide a comprehensive answer that addresses the query:"""
    )

    return {"relevant_docs": relevant_documents, "llm_response": response}


def summarize(query, limit):
    hs = HybridSearch(load_movies())
    relevant_documents = hs.rrf_search(query, k=60, limit=limit)
    response = generate_response(
        f"""
        Provide information useful to this query by synthesizing information from multiple search results in detail.
        The goal is to provide comprehensive information so that users know what their options are.
        Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.
        This should be tailored to SilasStreaming users. SilasStreaming is a movie streaming service.
        Query: {query}
        Search Results:
        {json.dumps(relevant_documents)}
        Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
    )

    return {"relevant_docs": relevant_documents, "llm_response": response}


def citations(query, limit):
    hs = HybridSearch(load_movies())
    relevant_documents = hs.rrf_search(query, k=60, limit=limit)
    response = generate_response(
        f"""
        Answer the question or provide information based on the provided documents.

        This should be tailored to SilasStreaming users. SilasStreaming is a movie streaming service.

        If not enough information is available to give a good answer, say so but give as good of an answer as you can while citing the sources you have.

        Query: {query}

        Documents:
        {json.dumps(relevant_documents)}

        Instructions:
        - Provide a comprehensive answer that addresses the query
        - Cite sources using [1], [2], etc. format when referencing information
        - If sources disagree, mention the different viewpoints
        - If the answer isn't in the documents, say "I don't have enough information"
        - Be direct and informative

        Answer:"""
    )

    return {"relevant_docs": relevant_documents, "llm_response": response}


def question(question, limit):
    hs = HybridSearch(load_movies())
    relevant_documents = hs.rrf_search(question, k=60, limit=limit)
    response = generate_response(
        f"""
        Answer the user's question based on the provided movies that are available on SilasStreaming.

        This should be tailored to SilasStreaming users. SilasStreaming is a movie streaming service.

        Question: {question}

        Documents:
        {json.dumps(relevant_documents)}

        Instructions:
        - Answer questions directly and concisely
        - Be casual and conversational
        - Don't be cringe or hype-y
        - Talk like a normal person would in a chat conversation

        Answer:"""
    )

    return {"relevant_docs": relevant_documents, "llm_response": response}
