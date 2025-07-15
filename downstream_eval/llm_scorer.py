import os
import json

from openai import OpenAI
from itertools import combinations

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY").strip()
if not OPENAI_API_KEY or not OPENAI_API_KEY.strip():
    raise ValueError("Missing OPENAI_API_KEY. Please set it as an environment variable.")
client = OpenAI(api_key=OPENAI_API_KEY)


def relevance_score(question, docs):
    
    formatted_docs = "\n".join(
        [f"({i+1}) {a}" for i, a in enumerate(docs)]
    )

    prompt = (
        "You are a scoring system, you have to evaluate the quality of each answer according to the given question.\n"
        f"Question: {question}, Answer list: {formatted_docs}\n"
        "Reply only [score1, score2, score3, ...]\n"
        "For example : [5, 1, 1, 1, 1, 1, 1, 1, 1, 1]\n"
    )

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0
    )

    content = response.choices[0].message.content.strip()
    try:
        score_list = json.loads(content)
        return score_list
    except Exception as e:
        print(f"Failed to parse {content} in JSON format")
        return []


def pairwise_score(question, docs):

    assert(len(docs) == 5)

    win_counts = [0] * 5
    for i, j in combinations(range(5), 2):
        prompt = (
            f"Query: {question}\n\n"
            f"Document A (Index {i}):\n{docs[i]}\n\n"
            f"Document B (Index {j}):\n{docs[j]}\n\n"
            "Which document is more relevant to the query? "
            "Respond with 'A' if Document A is more relevant, or 'B' if Document B is more relevant."
        )

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a helpful assistant. When presented with two documents and a query, "
                        "respond with 'A' or 'B' only, indicating which document is more relevant."
                    )
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.0
        )

        choice = response.choices[0].message.content.strip().upper()
        if choice.startswith("A"):
            win_counts[i] += 1
        elif choice.startswith("B"):
            win_counts[j] += 1
        else:
            continue

    sorted_indices = sorted(range(5), key=lambda idx: win_counts[idx], reverse=True)
    ranks = [0] * 5
    for rank_position, idx in enumerate(sorted_indices, start=1):
        ranks[idx] = rank_position

    reciprocal_scores = [1.0 / r for r in ranks]
    total = sum(reciprocal_scores)
    return [s / total for s in reciprocal_scores]

