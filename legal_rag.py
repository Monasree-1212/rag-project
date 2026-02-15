import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


data = open("legal_data.txt").read().split("###")
data = [x.strip() for x in data if x.strip()]

vec = TfidfVectorizer()
mat = vec.fit_transform(data)

while True:
    q = input("\nAsk legal query (exit to quit): ")
    if q == "exit":
        break

    qv = vec.transform([q])
    scores = cosine_similarity(qv, mat).flatten()
    best = data[scores.argmax()]

    prompt = f"Explain legal clause:\n{best}\nQuestion:{q}"

    res = ollama.chat(model="phi3", messages=[{"role":"user","content":prompt}])
    print(res["message"]["content"])
