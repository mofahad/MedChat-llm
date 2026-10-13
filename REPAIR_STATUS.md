# Local repair status

The original notebook is a retrieval and generation experiment using BioMistral-7B, MiniLM embeddings, FAISS, Flask, and React. Historical research claims are retained separately in `HISTORICAL_NOTES.md`. Fine-tuning, comparison scores, and reviewer-model results are not verified by the checked-in execution code.

The supported lightweight backend loads the supplied 16,407-row question-answer CSV and returns retrieved records on CPU. CSV loading and chat smoke checks pass. Eight backend tests, three frontend behavior tests, and the frontend production build pass. These are functionality checks, not a retrieval-quality or clinical benchmark. Optional local-model generation is tested with stubs for success/fallback labeling; an actual model has not been executed. Dataset provenance defaults to unverified supplied CSV until its source is documented.

## Repairs prepared

- Corrected undefined `df` and `row_to_text` references to `df1` and `row_to_text1`.
- Replaced the expired frontend tunnel with a configurable local backend URL.
- Removed embedded credentials and cleared saved notebook outputs.
- Added request validation, stateless API calls, and the missing clear-chat endpoint.
- Disabled storage of user medical questions in a local log.
- Removed the prompt's claim to be a licensed doctor.
- Guarded negative FAISS result indices and removed unconditional compilation.
- Used automatic model placement instead of moving an already quantized model.

## Reproduction blockers

`train.csv` is not included. Its required columns are `Question` and `Answer`. The notebook also retrieves `FreedomIntelligence/medical-o1-reasoning-SFT` from Hugging Face. Obtain authorized data and document its provenance before executing. Model weights and datasets require downloads even though there is no paid API.

The original 4-bit model path needs a compatible GPU/software environment. That notebook's model inference, hardware benchmark, and clinical evaluation remain unvalidated. Static parsing is only a syntax check.

## Frontend

In `frontend`, run `npm ci` and `npm start`. The default API URL is `http://127.0.0.1:5005`; `.env.example` documents the override. Start `backend.py` using the root README's environment configuration. The interface uses single-turn requests; the clear button resets local messages.

The existing dependency lockfile's npm audit reported 103 findings (6 low, 15 moderate, 78 high, 4 critical). This is inherited dependency debt, not a verified exploit assessment. Dependency upgrades need separate validation before public deployment; no force upgrade was applied.

Never commit tokens. If using ngrok, supply `NGROK_AUTHTOKEN` through the environment. The previously committed token must be revoked in the provider account; this repair does not rewrite Git history.
