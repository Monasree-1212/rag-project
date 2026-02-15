import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load(file):
    return [x.strip() for x in open(file).read().split("###") if x.strip()]


tickets = load("support_data.txt")

vec = TfidfVectorizer()
mat = vec.fit_transform(tickets)

while True:
    q = input("\nCustomer issue (exit to quit): ")
    if q == "exit":
        break

    qv = vec.transform([q])
    score = cosine_similarity(qv, mat).flatten()
    context = tickets[score.argmax()]

    prompt = f"Suggest solution:\n{context}\nIssue:{q}"

    res = ollama.chat(model="phi3", messages=[{"role":"user","content":prompt}])
    print(res["message"]["content"])
