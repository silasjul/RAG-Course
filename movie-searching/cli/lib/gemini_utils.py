from google import genai
from google.genai import errors
import os
from dotenv import load_dotenv
import time
import json

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)
model = "gemini-2.5-flash-lite"


def enhance_query(query: str, method: str):
    for attempt in range(3):
        try:
            match method:
                case "spell":
                    response = client.models.generate_content(
                        model=model,
                        contents=f"""
                        Fix any spelling errors in this movie search query.

                        Only correct obvious typos. Don't change correctly spelled words.

                        Query: "{query}"

                        If no errors, return the original query.
                        Corrected:""",
                    )
                case "rewrite":
                    response = client.models.generate_content(
                        model=model,
                        contents=f"""
                        Rewrite this movie search query to be more specific and searchable.

                        Original: "{query}"

                        Consider:
                        - Common movie knowledge (famous actors, popular films)
                        - Genre conventions (horror = scary, animation = cartoon)
                        - Keep it concise (under 10 words)
                        - It should be a google style search query that's very specific
                        - Don't use boolean logic

                        Examples:

                        - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                        - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                        - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"

                        Rewritten query:""",
                    )
                case "expand":
                    response = client.models.generate_content(
                        model=model,
                        contents=f"""
                        Expand this movie search query with related terms.

                        Add synonyms and related concepts that might appear in movie descriptions.
                        Keep expansions relevant and focused.
                        This will be appended to the original query.

                        Examples:

                        - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                        - "action movie with bear" -> "action thriller bear chase fight adventure"
                        - "comedy with bear" -> "comedy funny bear humor lighthearted"

                        Query: "{query}"
                        """,
                    )
            break
        except errors.ClientError as e:
            if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                print(
                    f"Rate limit hit in enhance_query, retrying in 10 seconds... ({attempt+1}/3)"
                )
                time.sleep(10)
            else:
                raise
    else:
        raise Exception("Max retries exceeded for enhance_query")
    print(f"Enhanced query ({method}): '{query}' -> '{response.text}'\n")
    return response.text


def rerank(
    query,
    results,
    method,
):
    match method:
        case "individual":
            reranked_results = []
            for r in results:
                for attempt in range(3):
                    try:
                        response = client.models.generate_content(
                            model=model,
                            contents=f"""
                            Rate how well this movie matches the search query.

                            Query: "{query}"
                            Movie: {r["document"]["title"]} - {r["document"]["description"]}

                            Consider:
                            - Direct relevance to query
                            - User intent (what they're looking for)
                            - Content appropriateness

                            Rate 0-10 (10 = perfect match).
                            Give me ONLY the number in your response, no other text or explanation.

                            Score:""",
                        )
                        break
                    except errors.ClientError as e:
                        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                            print(
                                f"Rate limit hit, retrying in 10 seconds... ({attempt+1}/3)"
                            )
                            time.sleep(10)
                        else:
                            raise
                else:
                    raise Exception("Max retries exceeded for individual rerank")
                try:
                    r["rerank_score"] = float(response.text)
                except ValueError as e:
                    print(f"Failed to parse score as float: {e}")
                    print(f"Response text: '{response.text}'")
                    raise
                reranked_results.append(r)
                time.sleep(3)
            return sorted(
                reranked_results, key=lambda key: key["rerank_score"], reverse=True
            )
        case "batch":
            id_to_result = {r["document"]["id"]: r for r in results}
            doc_list_str = json.dumps(results)
            for attempt in range(3):
                try:
                    response = client.models.generate_content(
                        model=model,
                        contents=f"""
                        Rank these movies by relevance to the search query.

                        Query: "{query}"

                        Movies:
                        {doc_list_str}

                        Return ONLY the document IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:

                        [75, 12, 34, 2, 1]
                        """,
                    )
                    break
                except errors.ClientError as e:
                    if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                        print(
                            f"Rate limit hit, retrying in 10 seconds... ({attempt+1}/3)"
                        )
                        time.sleep(10)
                    else:
                        raise
            else:
                raise Exception("Max retries exceeded for batch rerank")
            try:
                rerankings = json.loads(response.text)
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON response: {e}")
                print(f"Response text: '{response.text}'")
                raise
            print(rerankings)
            reranked_results = []
            for idx, id in enumerate(rerankings, 1):
                r = id_to_result[id]
                r["rerank_rank"] = idx
                reranked_results.append(r)
            return reranked_results
