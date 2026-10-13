# Medical source retrieval prototype

A local React and Flask prototype with a CPU-friendly retrieval backend. The default path uses TF-IDF lexical retrieval and displays source records. It does not silently simulate an LLM or pretend that retrieval scores establish correctness.

## Free local setup

Use Python 3.11 or 3.12. No paid API is required.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements-local.txt
$env:APP_DOMAIN="medical"
$env:MEDICAL_CSV="C:\path\to\data.csv"
$env:PORT="5005"
python backend.py
```

On macOS/Linux set the same environment variables with `export`. `.env.example` is documentation; the backend does not automatically load it.

The medical CSV needs `Question` and `Answer` columns. Source records are retrieved by question text. Datasets are supplied separately and are not committed. Document provenance, permissions, and freshness before sharing data or reporting real-world findings.

Start the frontend in another terminal:

```bash
cd frontend
npm ci
npm start
```

The frontend defaults to `http://127.0.0.1:5005`. Set `REACT_APP_API_URL` to change it. Local CORS allows only localhost frontend origins. The backend binds to localhost, stores no conversation history, and reports loaded corpus sizes at `/health`.

## Optional local generation

With Ollama running locally and a model already downloaded, set `OLLAMA_MODEL` to that model name. The backend sends retrieved excerpts to `http://127.0.0.1:11434/api/generate`. Generated responses are marked `local_llm_unverified`. If generation fails, it returns source records with an explicit fallback message. Generated summaries have not been validated in this repair pass.

## Tests and validation

```bash
python -m unittest test_backend -v
```

Seven backend tests cover ranking, provenance, abstention, request validation, missing data, stateless clearing, local CORS, and labeled generation/fallback responses. Real CSV loading and retrieval smoke checks are tracked in `REPAIR_STATUS.md`. These checks are not retrieval-quality benchmarks. Three frontend behavior tests and the production build pass locally. Actual local LLM execution remains pending.

## Original notebook

The original notebook preserves the transformer/FAISS experiment. It has separate hardware and data requirements and is not the lightweight startup path. `REPAIR_STATUS.md` documents its repaired bugs and remaining limitations. `HISTORICAL_NOTES.md` preserves earlier descriptions without treating unsupported measurements as verified results.

This educational prototype is not a diagnostic or treatment system and has no clinical validation.
