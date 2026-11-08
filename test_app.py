import json
from pathlib import Path
import tempfile
import unittest

from app import Retriever, create_app


class SourceExplorerTests(unittest.TestCase):
    def setUp(self):
        self.client = create_app().test_client()

    def test_model_question_returns_traceable_source(self):
        response = self.client.post('/chat', json={'question': 'What model does the notebook use?'})
        self.assertEqual(response.status_code, 200)
        body = response.get_json()
        self.assertEqual(body['sources'][0]['id'], 'model')
        self.assertIn('BioMistral', body['reply'])
        self.assertIn('[model]', body['reply'])
        self.assertEqual(body['mode'], 'lexical_retrieval')

    def test_unknown_query_abstains_without_inventing_answer(self):
        body = self.client.post('/chat', json={'question': 'xyzzabc qwervv'}).get_json()
        self.assertEqual(body['mode'], 'abstained')
        self.assertEqual(body['sources'], [])

    def test_invalid_requests_return_actionable_errors(self):
        for value in ([], {}, {'question': None}, {'question': ''}, {'question': ' '*5}, {'question': 'a'*2001}):
            self.assertEqual(self.client.post('/chat', json=value).status_code, 400)
        self.assertEqual(self.client.post('/chat', data='not json', content_type='application/json').status_code, 400)
        self.assertEqual(self.client.post('/chat', data='x'*40000, content_type='application/json').status_code, 413)

    def test_health_and_browser_demo(self):
        self.assertFalse(self.client.get('/health').get_json()['generation'])
        with self.client.get('/') as response:
            self.assertIn(b'RESEARCH SOURCE EXPLORER', response.data)

    def test_clear_does_not_change_another_clients_results(self):
        before = self.client.post('/chat', json={'question': 'FAISS retrieval'}).get_json()
        self.assertFalse(self.client.delete('/clear-chat').get_json()['stored_history'])
        after = create_app().test_client().post('/chat', json={'question': 'FAISS retrieval'}).get_json()
        self.assertEqual(before, after)

    def test_only_known_local_frontend_origins_get_cors(self):
        self.assertNotIn('Access-Control-Allow-Origin', self.client.get('/health', headers={'Origin':'https://example.com'}).headers)
        response = self.client.options('/chat', headers={'Origin':'http://localhost:3000'})
        self.assertEqual(response.headers['Access-Control-Allow-Origin'], 'http://localhost:3000')

    def test_custom_corpus_and_duplicate_validation(self):
        row = {'id':'example','title':'Synthetic example','text':'violet triangle demonstration','source':'test fixture'}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'notes.json'
            path.write_text(json.dumps([row]), encoding='utf-8')
            body = create_app(path).test_client().post('/chat', json={'question':'violet triangle'}).get_json()
            self.assertEqual(body['sources'][0]['id'], 'example')
        for records in ([], [row, row], [{}], [None]):
            with self.assertRaises(ValueError):
                Retriever(records)


if __name__ == '__main__':
    unittest.main()
