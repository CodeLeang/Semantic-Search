from flask import Flask, request, render_template
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)

class SemanticSearchEngine:
    def __init__(self, filepath):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.corpus = self._load_corpus(filepath)
        self.corpus_embeddings = self.model.encode(self.corpus, convert_to_tensor=True)

    def _load_corpus(self, filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines

    def search(self, query, top_k=5):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        results = util.semantic_search(query_embedding, self.corpus_embeddings, top_k=top_k)[0]
        return [(self.corpus[result['corpus_id']], result['score']) for result in results]

search_engine = SemanticSearchEngine('news.txt')

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        query = request.form['query']
        results = search_engine.search(query, top_k=10)
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
