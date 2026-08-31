# Outline Contract

## Section Tree

| # | Section | Intent | Citation Quota | Figure/Table Quota |
|---|---------|--------|---------------|--------------------|
| 1 | Introduction | Motivate the problem of Vietnamese financial statement QA; articulate why text-to-pandas is preferable to text-to-SQL/DSL for this domain; state the system's four main contributions; preview paper structure | 8–12 | 1 (architecture overview figure) |
| 2 | Related Work | Position against: (a) Financial QA benchmarks (FinQA, TAT-QA, ViFinQA, FinanceBench), (b) Text-to-SQL vs text-to-code approaches, (c) LLM code generation with self-correction, (d) RAG and hybrid retrieval for table QA, (e) Sandboxed code execution for LLM agents | 15–25 | 0–1 (optional comparison table) |
| 3 | System Architecture | Present the end-to-end pipeline as a state-machine workflow; describe each stage using logical component names (not filenames); cover offline pipeline (OCR extraction, entity classification, taxonomy induction, indexing) and online pipeline (query routing, hybrid retrieval, reranking, concept-coverage selection, grounded planning, AST-governed code synthesis, sandboxed execution, provenance construction) | 10–15 | 3–4 (pipeline architecture diagram, state graph topology, concept coverage matrix example, sandbox execution protocol) |
| 3.1 | Offline Data Pipeline | OCR table extraction, entity domain classification (4 Vietnamese regulatory domains), table type classification, normalization, taxonomy induction, semantic enrichment, metadata cataloging, vector indexing with blue/green deployment | 3–5 | 1 (offline pipeline diagram or dataset statistics table) |
| 3.2 | Query Routing & Metadata Parsing | Deterministic metadata extraction (ticker, year, report type), filter validation, semantic query construction | 2–3 | 0 |
| 3.3 | Multi-Stage Table Retrieval | Hybrid dense+BM25 search with RRF, per-ticker quota balancing, report-type fallback; cross-encoder reranking with BGE-M3; CSV context construction | 4–6 | 1 (retrieval funnel/progressive filtering diagram) |
| 3.4 | Concept-Coverage Table Selection | Parallel LLM scouts, lexical rescue, final arbiter with concept coverage matrix (accounting roles), coverage-locked buckets, correction sub-loop | 3–5 | 1 (concept coverage matrix example table) |
| 3.5 | Grounded Code Planning & AST-Governed Synthesis | Planning inventory construction, row hydration, LLM code generation with JSON schema enforcement, AST normalization passes (selector normalization, unit scale validation, rounding control, metadata isolation), evidence variable tracking | 4–6 | 1 (AST validation pipeline or code generation example) |
| 3.6 | Sandboxed Execution & Answer Construction | MicroVM isolation, DataFrame serialization protocol, bounded JSON result, provenance assembly, retry loop with structured feedback | 2–4 | 0–1 |
| 4 | Experimental Setup | Describe ViFinQA benchmark, evaluation metrics (answer accuracy, execution accuracy, evidence F2/precision/recall/MRR@5), baseline configurations, model specifications, ablation configurations | 5–8 | 1–2 (dataset statistics table, evaluation metrics table) |
| 5 | Results & Analysis | Main results on ViFinQA; ablation studies (removing metadata routing, removing concept coverage, removing AST validation, removing reranking, single-stage vs multi-stage retrieval); error analysis and case studies | 8–12 | 3–5 (main results table, ablation tables, error analysis figures, case study examples) |
| 6 | Discussion | Interpret results; discuss text-to-pandas vs text-to-SQL trade-offs with evidence; limitations (benchmark scope, LLM dependency, OCR quality sensitivity); generalization potential to other financial QA tasks and languages | 5–8 | 0–1 |
| 7 | Conclusion | Summarize contributions; restate key findings; outline future work directions (multimodal document understanding, cross-lingual transfer, interactive QA, fine-tuning) | 3–5 | 0 |

## Section-Specific Intent Notes

### Introduction
- Open with the challenge: answering quantitative financial questions requires locating specific tables across multi-page annual reports, understanding accounting structure, and performing correct numerical computation.
- Frame the text-to-pandas choice: explain why pandas code generation is better suited than SQL for OCR-extracted financial tables with heterogeneous schemas.
- List 4 contributions: (1) the end-to-end system, (2) concept coverage matrix for table selection, (3) AST-governed code synthesis with domain semantic validation, (4) evaluation on ViFinQA with full ablation.
- Preview paper organization.

### Related Work
- Structure as: Financial QA Benchmarks → Text-to-Code for Data Analysis → Table Retrieval & RAG → LLM Code Generation & Self-Correction → Secure Code Execution.
- Explicitly contrast with FinStat2SQL (Vietnamese text-to-SQL), PandasAI (general-purpose), LINX (goal-oriented exploration), Text-to-Pipeline (deterministic compilation).

### System Architecture
- Use logical component names throughout (e.g., "Query Router," "Table Retriever," "Cross-Encoder Reranker," "Concept Coverage Selector," "Code Planner," "AST Validator," "Execution Sandbox").
- Never reference source code filenames.
- Provide enough detail for replication.

### Results
- Report all metrics with confidence intervals or multiple-run statistics where feasible.
- Ablation table must show additive/subtractive contribution of each pipeline stage.
- Error analysis should categorize failures: retrieval miss, selection miss, code syntax error, semantic error (unit confusion, wrong row), execution timeout.

## Totals
- Target citations: 35–55
- Target figures/tables: 8–12
- Target pages: 8–10 (main body) + 2–4 (appendix with prompt templates and additional examples)
