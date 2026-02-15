import ollama
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def read_docs(file):
    with open(file, "r", encoding="utf-8") as f:
        return [x.strip() for x in f.read().split("###") if x.strip()]


class TechFinder:
    def __init__(self, docs):
        self.docs = docs
        self.tf = TfidfVectorizer(stop_words="english")
        self.mat = self.tf.fit_transform(docs)

    def get(self, q):
        v = self.tf.transform([q])
        scores = linear_kernel(v, self.mat).flatten()

        results = []
        for i in scores.argsort()[::-1][:2]:
            if scores[i] > 0:   # filter irrelevant results
                results.append(self.docs[i])

        return results


docs = read_docs("tech_data.txt")
engine = TechFinder(docs)

while True:
    q = input("\nAsk technical question (exit to quit): ")

    if q.lower() == "exit":
        break

    context_list = engine.get(q)

    if not context_list:
        print("No relevant documentation found.")
        continue

    context = "\n".join(context_list)

    system_prompt = (
        "You are a technical documentation assistant.\n"
        "Answer ONLY from the given documentation.\n"
        "If answer not found say: Not available in documentation."
    )

    try:
        res = ollama.chat(
            model="phi3",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Documentation:\n{context}\n\nQuestion:{q}"}
            ]
        )

        print("\nAnswer:")
        print(res["message"]["content"])

    except Exception as e:
        print("Error connecting to Ollama:", e)
