# Innovation Candidates

## Candidate 1: Deterministic-Metadata-Routed Text-to-Pandas Architecture for Financial Statement QA

- **1-Sentence Claim**: An end-to-end text-to-pandas pipeline that bounds LLM search through deterministic metadata routing and multi-stage progressive filtering (hybrid retrieval → cross-encoder reranking → concept-coverage-matrix selection → AST-governed code synthesis → sandboxed execution) achieves competitive accuracy on Vietnamese financial statement QA while providing full answer provenance.
- **Why It Matters**: Existing financial QA systems either use text-to-SQL (limited for file-based heterogeneous table structures), or treat code generation as a single-shot task without grounded planning or deterministic search bounding. The combination of domain-bounded retrieval with programmatic pandas code generation is unexplored for Vietnamese financial data.
- **Existing Coverage**: FinQA and TAT-QA introduced program-based and hybrid table+text reasoning. FinStat2SQL proposed multi-agent text-to-SQL for Vietnamese financial data. PandasAI enables natural-language-to-pandas for general data analysis. No prior work combines deterministic metadata routing, multi-stage table distillation, AST-governed code synthesis, and sandboxed execution in an integrated financial QA system.
- **Gap**: No end-to-end system that (a) generates pandas code instead of SQL/DSL programs for financial QA, (b) bounds retrieval through validated accounting metadata before search, and (c) provides traceable evidence provenance (source documents, table references, CSV paths) for every answer.
- **Evidence Needed**: System description; ViFinQA benchmark accuracy; ablation studies on pipeline stages; comparison with text-to-SQL baselines.
- **Falsification**: The system fails to produce accurate answers on ViFinQA, or simpler approaches (e.g., single-stage retrieval + direct code generation) match its accuracy.

**VERDICT: KEEP** — Strong primary claim for a system proposal paper. Clear gap, falsifiable, full evidence path available.

---

## Candidate 2: Concept Coverage Matrix for Accounting-Role-Aware Table Selection

- **1-Sentence Claim**: A two-stage LLM selection mechanism with an explicit concept coverage matrix (requiring accounting roles such as direct, numerator, denominator, beginning_balance, ending_balance, comparison_operand) significantly improves table selection precision over standard reranking for multi-table financial calculations.
- **Why It Matters**: Financial questions often require synthesizing data across multiple tables (e.g., computing ratios from balance sheet and income statement, comparing year-over-year values). Standard semantic reranking cannot reason about computational completeness.
- **Existing Coverage**: Cross-encoder rerankers (BGE-M3) are standard for precision improvement. Table selection for QA typically uses top-k reranking without computational completeness verification. No prior work models table selection as a concept coverage optimization problem with accounting role constraints.
- **Gap**: No existing table selection method enforces that the selected table set covers all required computational operands with specific accounting roles.
- **Evidence Needed**: Evidence retrieval metrics (F2, precision, recall, MRR@5); ablation removing the concept coverage matrix; comparison of scout+arbiter vs. top-k reranking.
- **Falsification**: Removing the concept coverage matrix does not degrade selection quality on multi-table questions.

**VERDICT: KEEP** — Strong secondary claim. Novel mechanism with clear ablation path.

---

## Candidate 3: AST-Governed Code Synthesis with Domain Semantic Validation

- **1-Sentence Claim**: Combining AST-level code linting with domain-specific semantic validation (unit scale normalization for VND/million/billion, rounding control, metadata isolation guards) reduces LLM code generation errors in financial calculations compared to unconstrained generation.
- **Why It Matters**: LLMs frequently produce mathematically plausible but semantically incorrect financial calculations — confusing VND vs. million VND storage conventions, applying unauthorized rounding, or fabricating metadata attributes. These errors are invisible to syntax-only validation.
- **Existing Coverage**: LLM code self-correction via execution feedback is established (LangGraph, ReAct). Static analysis of LLM-generated code is explored in software engineering. No prior work applies AST-level domain-specific semantic validation (unit scales, rounding rules, metadata isolation) to financial code generation.
- **Gap**: Existing code generation systems validate syntax and execution success but do not validate financial semantic correctness at the AST level.
- **Evidence Needed**: Error categorization study; ablation removing AST validation passes; comparison of error rates with/without semantic validation.
- **Falsification**: AST validation does not reduce error rates compared to raw LLM generation with execution feedback alone.

**VERDICT: KEEP** — Strong secondary claim. Novel technical contribution with clear evidence path.

---

## Candidate 4: Zero-Trust Sandboxed Execution with Bounded Result Protocol for Financial QA

- **1-Sentence Claim**: Executing LLM-generated pandas code in isolated Firecracker microVMs with network isolation and a bounded (≤1KB) JSON numeric scalar result protocol ensures security without sacrificing execution throughput.
- **Why It Matters**: Untrusted code execution is a known security risk in LLM agent systems.
- **Existing Coverage**: E2B and Firecracker microVM isolation are well-established infrastructure components. Docker/gVisor sandboxing is standard practice.
- **Gap**: The specific integration pattern is novel, but the core contribution is engineering, not research.
- **Evidence Needed**: Security analysis; execution latency measurements.
- **Falsification**: N/A — this is a design practice, not a falsifiable claim.

**VERDICT: REJECT as standalone claim** — Engineering contribution, not a novel research claim. Describe as a design decision within the system architecture section.

---

## Candidate 5: Taxonomy Induction from OCR Vietnamese Financial Statements

- **1-Sentence Claim**: Inducing accounting taxonomies from OCR-extracted financial statements across four Vietnamese regulatory entity domains (DN, TCTD, CTCK, BH) improves retrieval quality by providing semantically rich index text without requiring expensive LLM annotation.
- **Why It Matters**: Vietnamese financial statements follow domain-specific accounting standards (VAS Circulars 200, 49, 210/334, 232) with different chart-of-account layouts per entity type.
- **Existing Coverage**: Financial taxonomy standards exist (XBRL), but no prior work on automatic taxonomy induction from OCR Vietnamese financial statements.
- **Gap**: Narrow domain-specific contribution; complements the primary system claim.
- **Evidence Needed**: Coverage statistics; retrieval quality comparison with/without taxonomy enrichment.
- **Falsification**: Taxonomy-enriched index text does not improve retrieval recall.

**VERDICT: KEEP as secondary/supporting claim** — Too narrow for primary, but strengthens the offline pipeline contribution.

---

## Summary

| # | Candidate | Verdict | Role |
|---|-----------|---------|------|
| 1 | Deterministic-Metadata-Routed Text-to-Pandas Architecture | **KEEP** | Primary claim |
| 2 | Concept Coverage Matrix for Table Selection | **KEEP** | Secondary claim |
| 3 | AST-Governed Code Synthesis with Semantic Validation | **KEEP** | Secondary claim |
| 4 | Zero-Trust Sandboxed Execution | **REJECT** | Design decision (not a research claim) |
| 5 | Taxonomy Induction from Vietnamese OCR Statements | **KEEP** | Supporting claim |
