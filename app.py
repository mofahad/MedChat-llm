"""Free local source-retrieval demo. No model, token, or network call is hidden here."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import re

from flask import Flask, jsonify, request, send_from_directory

ROOT = Path(__file__).resolve().parent
STOPWORDS = frozenset('a an the is are was what which how does do in on for of to and with it this that'.split())


def tokenize(text: str) -> list[str]:
    return [word for word in re.findall(r'[a-z0-9]+', text.lower()) if word not in STOPWORDS]


class Retriever:
    """Small deterministic TF-IDF/cosine baseline with explicit abstention."""

    def __init__(self, records: list[dict]):
        if not isinstance(records, list) or not records:
            raise ValueError('The corpus must be a nonempty JSON list.')
        self.records = []
        seen = set()
        for record in records:
            if not isinstance(record, dict) or any(
                not isinstance(record.get(key), str) or not record[key].strip()
                for key in ('id', 'title', 'text', 'source')
            ):
                raise ValueError('Each record needs nonempty id, title, text, and source strings.')
            if record['id'] in seen:
                raise ValueError('Corpus record IDs must be unique.')
            seen.add(record['id'])
            self.records.append({key: record[key] for key in ('id', 'title', 'text', 'source')})
        counts = [Counter(tokenize(row['title'] + ' ' + row['text'])) for row in self.records]
        frequencies = Counter(word for document in counts for word in document)
        self.idf = {word: math.log((1 + len(counts)) / (1 + count)) + 1 for word, count in frequencies.items()}
        self.vectors = [self._vector(document) for document in counts]

    def _vector(self, counts: Counter) -> dict[str, float]:
        vector = {word: count * self.idf[word] for word, count in counts.items() if word in self.idf}
        norm = math.sqrt(sum(value * value for value in vector.values()))
        return {word: value / norm for word, value in vector.items()} if norm else {}

    def search(self, question: str, top_k: int = 3) -> list[dict]:
        query = self._vector(Counter(tokenize(question)))
        scored = [(sum(value * vector.get(word, 0) for word, value in query.items()), index)
                  for index, vector in enumerate(self.vectors)]
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        return [{**self.records[index], 'score': round(score, 4)}
                for score, index in scored[:top_k] if score > 0.05]


def create_app(corpus_path: str | Path | None = None) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config['MAX_CONTENT_LENGTH'] = 32 * 1024
    path = Path(corpus_path) if corpus_path else ROOT / 'data' / 'demo_notes.json'
    retriever = Retriever(json.loads(path.read_text(encoding='utf-8')))

    @app.after_request
    def response_headers(response):
        origin = request.headers.get('Origin')
        if origin in ('http://localhost:3000', 'http://127.0.0.1:3000'):
            response.headers['Access-Control-Allow-Origin'] = origin
            response.headers['Vary'] = 'Origin'
            response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
            response.headers['Access-Control-Allow-Methods'] = 'GET, POST, DELETE, OPTIONS'
        response.headers['X-Content-Type-Options'] = 'nosniff'
        response.headers['Cache-Control'] = 'no-store'
        return response

    @app.get('/')
    def home():
        return send_from_directory(ROOT / 'demo', 'index.html')

    @app.get('/health')
    def health():
        return jsonify(status='ok', mode='lexical_retrieval', documents=len(retriever.records), generation=False)

    @app.route('/chat', methods=['POST', 'OPTIONS'])
    def chat():
        if request.method == 'OPTIONS':
            return '', 204
        body = request.get_json(silent=True)
        question = body.get('question') if isinstance(body, dict) else None
        if not isinstance(question, str) or not question.strip() or len(question) > 2000:
            return jsonify(error='Send a nonempty question of at most 2,000 characters.'), 400
        sources = retriever.search(question.strip())
        reply = '\n\n'.join(f"[{row['id']}] {row['text']}" for row in sources)
        if not sources:
            reply = 'No matching source was found. Try a question about the model, retrieval, or project limitations.'
        return jsonify(reply=reply, sources=sources,
                       mode='lexical_retrieval' if sources else 'abstained',
                       notice='Source excerpts, not generated medical advice. Similarity is not confidence.')

    @app.route('/clear-chat', methods=['DELETE'])
    def clear_chat():
        # Chat state belongs to each browser. Never store shared patient/chat history.
        return jsonify(status='cleared', stored_history=False)

    @app.errorhandler(413)
    def too_large(_error):
        return jsonify(error='Request body is too large.'), 413

    return app


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, help='JSON records with id/title/text/source fields')
    parser.add_argument('--port', type=int, default=5005)
    args = parser.parse_args()
    try:
        application = create_app(args.corpus)
    except (OSError, ValueError) as error:
        parser.exit(2, f'Cannot load corpus: {error}\n')
    application.run(host='127.0.0.1', port=args.port, debug=False)
