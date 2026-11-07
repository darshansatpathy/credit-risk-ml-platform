# ADR-0001: Every model is monotone in declared directions

**Status:** accepted

## Context

An unconstrained model can learn that, in some region, more delinquency means less risk. It
may even improve AUC slightly. No credit officer, validator or customer would accept a decline
that a later payment made more likely.

## Decision

Each feature declares a direction. Boosted models use monotone constraints; the scorecard
merges bins and drops wrong-signed characteristics. The policy gate checks monotonicity
independently by sweeping each feature for 300 applicants.

## Consequences

- Reason codes are coherent: a feature that raised risk did so in the expected way.
- The planted-fault test shows the same LightGBM without constraints fails the check.
