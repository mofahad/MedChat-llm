import csv
from pathlib import Path
import tempfile
import unittest

from backend import create_app, load_csv


class ExternalMedicalSourceTests(unittest.TestCase):
    def test_supplied_qa_records_are_returned_with_provenance(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'fixture.csv'
            with path.open('w', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['Question', 'Answer'])
                writer.writeheader()
                writer.writerow({'Question':'What is the violet triangle?', 'Answer':'Synthetic source record A.'})
                writer.writerow({'Question':'What is the amber square?', 'Answer':'Synthetic source record B.'})
            corpus = load_csv(path, 'medical', provenance='synthetic')
            client = create_app({'medical':corpus}, mode='medical').test_client()
            body = client.post('/chat', json={'question':'violet triangle'}).get_json()
            self.assertEqual(body['sources'][0]['id'], 'medical:row:2')
            self.assertEqual(body['sources'][0]['provenance'], 'synthetic')
            self.assertIn('Synthetic source record A.', body['reply'])

    def test_invalid_csv_schema_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'bad.csv'
            path.write_text('Question\nexample\n', encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'required columns'):
                load_csv(path, 'medical')


if __name__ == '__main__':
    unittest.main()
