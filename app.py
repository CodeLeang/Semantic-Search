from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer, util
import torch

app = Flask(__name__)

class SemanticSearchEngine:
    def __init__(self, filepath):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.corpus = self._load_corpus(filepath)
        self.corpus_embeddings = self.model.encode(self.corpus, convert_to_tensor=True)

    def _load_corpus(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]

    def search(self, query, top_k=5, score_threshold=0.4):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        top_results = torch.topk(cos_scores, k=top_k)
        results = [
            (self.corpus[idx], float(score))
            for score, idx in zip(top_results.values, top_results.indices)
            if float(score) >= score_threshold
        ]
        return results

search_engine = SemanticSearchEngine('news.txt')

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    query = ""
    if request.method == 'POST':
        query = request.form.get('query', '')
        top_k = int(request.form.get('top_k', 5))
        score_threshold = float(request.form.get('score_threshold', 0.4))
        if query:
            results = search_engine.search(query, top_k=top_k, score_threshold=score_threshold)
    return render_template('index.html', results=results, query=query)

if __name__ == '__main__':
    app.run(port=5000, debug=True)
