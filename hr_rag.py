import ollama
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(file):
    with open(file, "r", encoding="utf-8") as f:
        data = f.read().split("###")

    return [d.strip() for d in data if d.strip()]


class HRSearch:
    def __init__(self, docs):
        self.docs = docs
        self.vec = TfidfVectorizer(stop_words="english")
        self.matrix = self.vec.fit_transform(docs)

    def search(self, query):
        q = self.vec.transform([query])
        score = cosine_similarity(q, self.matrix).flatten()
        return [self.docs[i] for i in score.argsort()[::-1][:2] if score[i] > 0]


def generate(context, question):

    system_prompt = (
        "You are an HR policy assistant.\n"
        "Answer ONLY from the provided company HR policies.\n"
        "Give short direct answers.\n"
        "Do NOT generate extra text.\n"
        "If answer not found say 'Not available in document.'"
    )

    user_prompt = f"""
Policies:
{context}

Question: {question}
"""

    res = ollama.chat(
        model="phi3",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        options={"temperature": 0.1}
    )

    return res["message"]["content"]


docs = load_data("hr_data.txt")
engine = HRSearch(docs)

while True:
    q = input("\nAsk HR question (exit to quit): ")

    if q.lower() == "exit":
        break

    result = engine.search(q)

    if not result:
        print("No policy found")
        continue

    print("\nAnswer:")
    print(generate("\n".join(result), q))
