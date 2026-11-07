# ADR-0003: Decide on the raw score; report the calibrated PD

**Status:** accepted

## Context

Isotonic calibration produced about 30 distinct PDs. A cutoff on the PD falls on one of those
steps, so it approved 76% of applicants against a policy of 80%.

## Decision

Set and apply the cutoff on the raw model score, which has no ties, and report the PD that
corresponds to it.

## Consequences

- Realized approval matches the policy (80.0% on validation, 80.6% on test).
- PD and decision can never disagree in order, because calibration is non-decreasing.
