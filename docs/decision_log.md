# Architectural Decision Log (ADR)

This document tracks major technical decisions, rationale, and alternatives considered for the Customer Intelligence Platform.

---

## ADR 1: Calibrated Probabilities for ML Model
- **Status:** Approved
- **Context:** Raw prediction scores from tree-based ensembles (such as Gradient Boosting) do not represent true probabilities because their predictions are pushed away from the extreme values.
- **Decision:** Wrap the `GradientBoostingClassifier` with `CalibratedClassifierCV(method="isotonic", cv=5)` to guarantee well-calibrated probabilities.
- **Rationale:** True probability outputs are required to accurately compute confidence intervals via bootstrapping and to map conversions into `LOW`, `MEDIUM`, and `HIGH` probability bands.
- **Alternatives:** Platt's scaling (sigmoid calibration) was considered, but Isotonic calibration performs better when there is sufficient calibration data.

---

## ADR 2: File-Based Serialized Metrics for Relative Gate Baseline
- **Status:** Approved
- **Context:** To evaluate improved models, we must compare their scores against a production baseline. Querying an external MLflow server during automated builds is fragile and introduces latency.
- **Decision:** Include model performance metrics directly inside the serialized model pickle file (`conversion_model.pkl`) inside the `metrics` dictionary.
- **Rationale:** This makes baseline comparison extremely fast and robust, as we can load the existing model directly from disk, fetch its validation scores, and verify thresholds before overwriting it.
- **Alternatives:** Querying MLflow database or using Git tags to look up baseline scores. This was rejected to avoid heavy external dependencies.

---

## ADR 3: Extractive RAG Refusal Threshold
- **Status:** Approved
- **Context:** When users ask questions outside the domain of complaints (e.g. asking for recipe instructions or general knowledge), standard RAG systems hallucinate or synthesize unrelated snippets.
- **Decision:** Enforce a similarity cutoff threshold of `0.35` on the top retrieved chunk. If the cosine similarity score is below this limit, the pipeline returns a standard refusal response and a confidence score of `0.0`.
- **Rationale:** Limits system vulnerability to out-of-domain and adversarial prompts while maintaining clean citations.
- **Alternatives:** Prompt engineering alone inside LLM systems, which is easily bypassed by jailbreaks.

---

## ADR 4: Unified Batch Scoring API
- **Status:** Approved
- **Context:** Clients need to perform inferences on large datasets.
- **Decision:** Implement a unified endpoint `/ml/batch-score` accepting either a JSON payload (list of Pydantic models) or a CSV file upload.
- **Rationale:** Streamlines bulk scoring without requiring the client to chunk payloads or convert formats. Gracefully handles row-level validation errors so that one bad record does not halt the entire batch.
