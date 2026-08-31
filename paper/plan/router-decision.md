# Router Decision

## Chosen Mode
empirical

## Rationale
The paper's primary contribution is a **novel system/method** — the FinLens
text-to-pandas pipeline — that makes specific design claims requiring
**experimental validation** on the ViFinQA benchmark. The core claims are:

1. **System-level**: The end-to-end pipeline achieves competitive accuracy on
   Vietnamese financial statement QA (requires benchmark evaluation).
2. **Component-level**: The concept coverage matrix improves table selection
   (requires ablation study). AST-governed code synthesis reduces errors
   (requires ablation study and error analysis).
3. **Design-level**: Text-to-pandas is more suitable than text-to-SQL for this
   domain (requires qualitative comparison with evidence from experiments).

All three levels require experiments, ablation studies, and quantitative metrics
to support — this is definitively an empirical paper.

The paper is **not** a survey/review because:
- It does not synthesize or organize prior work as its primary contribution.
- It proposes a specific, implementable system with novel components.
- Its claims are falsifiable through experiments.

## Alternative Path
A **review/survey** mode could have been chosen if the paper were framed as a
survey of text-to-code approaches for financial QA, with FinLens presented as a
case study. This was rejected because:
- The user explicitly asked for a "proposal paper describing the system."
- The system has enough novel components (concept coverage matrix, AST-governed
  synthesis, taxonomy induction) to warrant an empirical treatment.
- A survey would dilute the system's specific contributions.

The review path would look like: a taxonomy of text-to-SQL vs text-to-code
approaches for financial QA, with a systematic comparison of retrieval
strategies, code generation methods, and execution environments. FinLens would
be one entry in a comparison table. This is a valid paper but not what the user
requested.

## Handoff Date
2026-08-31

## Downstream Writer
Route to: `empirical-paper-writer` skill
(`../empirical-paper-writer/SKILL.md`)

## Handoff Artifacts Produced
- [x] `brief/topic-brief.md`
- [x] `brief/contribution-map.yaml`
- [x] `brief/evidence-matrix.csv`
- [x] `notes/innovation/candidates.md`
- [x] `plan/outline-contract.md`
- [x] `plan/router-decision.md` (this file)
