# Verification and limits

Repair checked on November 7, 2026 (America/Indianapolis).

## Evidence from the original checkout

- The only backend was a notebook with external dataset/model requirements; there was no standalone, documented local application.
- The React client used a hard-coded historical ngrok endpoint, swallowed request failures, and called a clear-chat route that the notebook did not implement.
- Dataset assembly referenced `df` and `row_to_text`, although the preceding definitions used `df1` and `row_to_text1`.
- The checked-in frontend test looked for the Create React App starter text.
- The notebook configures BioMistral-7B and FAISS IndexIVFFlat, not a packaged LLaMA/QLoRA/BGE/IVF-PQ deployment.
- No adapter checkpoint, held-out medical evaluation data, clinical evaluator script, GPU-memory measurement, or production-latency record is supplied.

## Checks actually completed

- Started from a fresh Git clone of the default branch.
- Created an isolated Python 3.12.14 environment and installed Flask 3.1.2 from `requirements.txt`.
- Backend: seven tests passed, covering traceable excerpts, abstention, malformed/oversized requests, health and page serving, stateless clearing, local-only CORS, and custom corpus validation.
- Consolidation: preserved the earlier CSV backend and notebook repairs. Installed its scikit-learn dependency in the same isolated environment; all seventeen backend checks pass, including the original eight tests and two new medical-CSV schema/provenance checks using synthetic fixtures. No real medical dataset or local LLM was executed.
- Notebook: sixteen Python code cells parse after excluding notebook shell commands; saved outputs are absent and a targeted secret-literal scan passes. This is static validation, not GPU inference. Embedded credentials removed by the earlier repair stay removed.
- Ran `npm ci` for the existing React client with Node 22.14.0.
- Frontend: three behavior tests passed, covering submission and citations, visible failure/retry, and local clearing.
- React production build completed successfully. The existing Browserslist dataset reports that it is old; no dependency modernization or security audit is claimed.
- Browser: exercised model retrieval, retrieval inspection, and an unmatched query. The unmatched query explicitly abstained.
- The GIF captures the actual local browser demo; it is not a mockup.

On this Windows host the restricted process environment blocked some dependency/test-cache and temporary-file writes; those commands were rerun with ordinary host file access. This was an environment limitation, not a passing test result.

## What this does not establish

The demo retrieves implementation notes using a lexical baseline. It does not generate answers, validate clinical guidance, reproduce medical hallucination reduction, measure 7B inference, or establish any GPU/latency result. It is not a substitute for a held-out retrieval benchmark or a clinical safety evaluation.

The original notebook and its dependencies were not run end to end. GPU access, missing supplemental data, and absent adapter/benchmark artifacts remain limitations. The earlier draft's notebook fixes and removal of embedded credentials were preserved. Historical API credentials and tunnels were not tested or renewed, so their expiry status is unknown. Removing credentials from the current notebook does not remove them from Git history.

The repair is offered as a pull request. Until it is merged, the default branch will still contain the previous startup path. Check the branch/PR state before sharing a runnable link.
