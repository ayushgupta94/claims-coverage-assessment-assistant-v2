# ADR 0003: The LLM never decides eligibility, fraud, confidence or routing

**Status:** accepted

## Context
Claim decisions are regulated and disputed. An insurer must be able to say
*why* a claim was rejected, and the same claim must get the same answer.
LLM output varies, can be manipulated by text in the claim, and an LLM's
self-reported confidence is not calibrated.

## Decision
- Coverage rules (the claim type must be an insured peril of the product and
  not excluded; waiting period; sum insured) and fraud scoring are
  deterministic tools. Insured perils are an allow-list that fails closed: a
  claim type the wording doesn't insure is not covered, rather than passing
  because nothing excludes it.
- The LLM reads the wording, proposes an outcome, cites clauses and explains.
- `decision/policy.py` (pure functions) makes the final call: the rules win
  when the LLM contradicts them; only retrieved clauses may be cited;
  confidence is computed from evidence; a fixed list of conditions sends the
  claim to a human, with the reasons recorded.

## Consequences
- Decisions are reproducible and explainable; the policy is unit-tested. The
  fraud score only looks at claims filed before the one being scored, so
  recomputing it later gives the same answer.
- The LLM adds value where rules can't: reading wording that excludes a loss
  the claim type doesn't reveal ("disguised exclusions"). Those cases are
  escalated rather than auto-decided, which the evals measure.
- The confidence weights are a documented heuristic, not a trained calibration;
  the next step would be calibrating them against reviewer outcomes.
