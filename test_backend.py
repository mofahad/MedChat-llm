import unittest
from unittest.mock import patch
from backend import Corpus, create_app

class RetrievalTests(unittest.TestCase):
    def setUp(self):
        corpus = Corpus('housing', [
            {'id':'housing:1','text':'Bloomington Indiana apartment rent 900','excerpt':'Demo apartment in Bloomington'},
            {'id':'housing:2','text':'Portland Oregon house rent 1800','excerpt':'Demo house in Portland'},
        ])
        self.client = create_app({'housing':corpus}, mode='relocation').test_client()

    def test_ranking_and_sources(self):
        result = self.client.post('/chat', json={'question':'Bloomington apartment'}).json
        self.assertEqual(result['sources'][0]['id'], 'housing:1')
        self.assertEqual(result['mode'], 'tfidf_retrieval')

    def test_no_match(self):
        result = self.client.post('/chat', json={'question':'zzxyq'}).json
        self.assertEqual(result['mode'], 'abstained')

    def test_validation(self):
        for body in ([], {}, {'question':None}, {'question':''}, {'question':'a'*4001}):
            self.assertEqual(self.client.post('/chat', json=body).status_code, 400)

    def test_missing_data(self):
        client = create_app({}, mode='medical').test_client()
        self.assertEqual(client.get('/health').status_code, 503)
        self.assertEqual(client.post('/chat', json={'question':'Test'}).status_code, 503)

    def test_stateless_and_cors(self):
        self.assertEqual(self.client.delete('/clear-chat').status_code, 200)
        response = self.client.options('/chat', headers={'Origin':'http://localhost:3000'})
        self.assertEqual(response.headers['Access-Control-Allow-Origin'], 'http://localhost:3000')
        self.assertNotIn('Access-Control-Allow-Origin', self.client.get('/health', headers={'Origin':'https://example.com'}).headers)

    def test_generation_fallback_is_explicit(self):
        with patch.dict('os.environ', {'OLLAMA_MODEL':'test-local-model'}), patch('backend.local_generate', side_effect=OSError('unavailable')):
            result = self.client.post('/chat', json={'question':'Bloomington apartment'}).json
        self.assertEqual(result['mode'], 'tfidf_retrieval')
        self.assertIn('returned source records instead', result['generation_error'])

    def test_generated_summary_is_labeled(self):
        with patch.dict('os.environ', {'OLLAMA_MODEL':'test-local-model'}), patch('backend.local_generate', return_value='Fixture summary'):
            result = self.client.post('/chat', json={'question':'Bloomington apartment'}).json
        self.assertEqual(result['mode'], 'local_llm_unverified')
        self.assertIn('not independently verified', result['reply'])

if __name__ == '__main__':
    unittest.main()
