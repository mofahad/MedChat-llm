# Turning future claims into evidence

The runnable local source explorer provides a reproducible starting point. It does not make historical performance claims true. Report results only after collecting the following evidence.

## Hallucination or grounding improvements

Use a permitted, versioned medical source corpus and a held-out question set with source-linked reference answers. Separate training, retrieval, and evaluation splits to prevent answer leakage. Compare the same model and prompts with and without retrieval; define claim-level grounding and abstention criteria before testing. Save the actual responses, per-question judgments, evaluator instructions, failures, dataset hashes, and exact model/adapter revisions. Report absolute rates and relative change with denominators and uncertainty. An LLM judge alone is not clinical validation.

## Quantization memory savings

Provide the base model and trained adapter, then compare the intended precisions on the same GPU with the same prompt length, output limit, batching, and loading strategy. Record peak allocated/reserved memory, the complete environment, and output-quality checks. A quantization setting in a notebook is not a measured memory reduction.

## Inference latency

State whether latency means time to first token, model-only processing, or complete end-to-end generation. Benchmark a fixed workload after warm-up, under stated concurrency, and include failures and timeouts. Save individual observations and report the latency distribution alongside output lengths. Do not describe fast source lookup as 7B model inference.

## Model identity and ownership

The checked-in notebook uses BioMistral-7B. A later LLaMA/QLoRA implementation would need its own source, model/adapter revisions, training configuration, dataset provenance, evaluation records, and team contribution statement. A different model name must not be substituted into the README to match a resume.

## Free-first boundary

The current demo runs on CPU without keys or model downloads. No paid endpoint, GPU rental, credential renewal, or Hugging Face key is required for this repair. Model training and GPU benchmarking remain future work; do not promise a particular improvement before observing it.
