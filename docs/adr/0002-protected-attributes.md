# ADR-0002: Protected attributes are for testing outcomes, not inputs

**Status:** accepted

## Context

Sex, age and marital status are protected in most lending regimes; education can act as a
proxy. Excluding them from the inputs does not make outcomes fair by itself, so they still
have to be measured.

## Decision

Exclude sex, age, marital status and education from the features. Join them back to the test
decisions and report approval rates, adverse impact and equal opportunity by group. Gate on
adverse impact for sex and age.

## Consequences

- Approval differences that remain come from the inputs that are allowed (limits, delinquency,
  utilization) and are visible in the report.
- Age is the closest to the threshold and is called out for review at every retraining.
