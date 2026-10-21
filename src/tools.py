
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContextRetriever:
    """Retrieve relevant passages from a JSONL dataset using TF-IDF."""

    def __init__(self, data_path, top_k=3):
        self.data_path = Path(data_path)
        self.top_k = top_k

        with self.data_path.open("r", encoding="utf-8") as file:
            records = [
                json.loads(line)
                for line in file
                if line.strip()
            ]

        # Remove duplicate passages while preserving order.
        self.contexts = list(dict.fromkeys(
            record["context"].strip()
            for record in records
            if record.get("context", "").strip()
        ))

        if not self.contexts:
            raise ValueError(f"No usable contexts found in {self.data_path}")

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words="english",
        )
        self.context_vectors = self.vectorizer.fit_transform(self.contexts)

    def retrieve(self, question):
        """Return up to top_k relevant contexts with similarity scores."""
        query_vector = self.vectorizer.transform([question])
        scores = cosine_similarity(
            query_vector,
            self.context_vectors,
        ).flatten()

        ranked_indices = scores.argsort()[::-1]
        results = []

        for index in ranked_indices:
            if scores[index] <= 0:
                continue

            results.append({
                "context": self.contexts[index],
                "score": float(scores[index]),
            })

            if len(results) >= self.top_k:
                break

        return results
