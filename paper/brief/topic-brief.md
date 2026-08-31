# Topic Brief

## Topic
FinLens: An end-to-end text-to-pandas system for numerical question answering over Vietnamese corporate financial statements, featuring deterministic metadata routing, multi-stage table retrieval with concept coverage verification, AST-governed code synthesis, and sandboxed execution.

## Scope

### In Scope
- System architecture and design rationale for each pipeline stage (offline data preparation through online answer generation)
- Text-to-pandas code generation as an alternative to text-to-SQL/DSL for financial table reasoning
- Multi-stage progressive table retrieval: hybrid dense+lexical search → cross-encoder reranking → accounting-role-aware concept coverage selection
- AST-level code validation with domain-specific semantic checks (unit normalization, rounding control, metadata isolation)
- Sandboxed code execution in isolated microVMs with bounded result protocol
- Evaluation on the ViFinQA benchmark (Vietnamese financial statement QA)
- Evidence provenance and answer traceability
- Domain-specific offline pipeline: OCR table extraction, entity classification across Vietnamese accounting standards, taxonomy induction

### Out of Scope
- Arbitrary free-text financial question answering (system is benchmark-scoped)
- Training or fine-tuning of the underlying LLM (system uses off-the-shelf models)
- Multilingual generalization beyond Vietnamese financial statements
- Comparison with vision-language models for document understanding
- Real-time streaming or interactive conversational QA

## Audience
Researchers and practitioners in financial NLP, table-based question answering, LLM-based code generation, and retrieval-augmented generation (RAG) systems. Secondary audience: Vietnamese fintech/AI community and VLSP shared task participants.

## Constraints
- Venue: arXiv preprint (potential submission to ACL/EMNLP Findings, VLSP workshop, or IEEE Access)
- Page target: 8–10 pages (main body), plus appendix for prompt templates and additional ablations
- Deadline: No hard deadline; quality-first
- Special requirements:
  - Do not reference specific source code filenames in the paper text
  - All system descriptions should use logical component names and architectural abstractions
  - Experimental results on ViFinQA must be reproducible
  - Evidence provenance must be described with sufficient detail for replication

## Key Terms
- text-to-pandas
- financial question answering
- Vietnamese financial statements
- retrieval-augmented generation
- table retrieval
- hybrid search (dense + BM25)
- cross-encoder reranking
- concept coverage matrix
- AST code validation
- sandboxed code execution
- OCR table extraction
- accounting taxonomy induction
- ViFinQA benchmark
- LangGraph state machine
