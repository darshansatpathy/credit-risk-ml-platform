# Security policy

## Reporting a vulnerability

Please do not open a public issue for security problems. Use GitHub's
[private vulnerability reporting](https://github.com/darshansatpathy/credit-risk-ml-platform/security/advisories/new)
for this repository. You should get an acknowledgement within three working days.

## Operating it safely

- Models load from text and JSON only; never load model files from an untrusted source anyway.
- The scoring API has no authentication. Put it behind your gateway.
- Applications are validated before scoring and rejected, not scored, when they fail.
- Scores and reasons are decision support. Lending decisions remain subject to the policies and
  regulations that apply to you.
