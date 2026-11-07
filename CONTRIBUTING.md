# Contributing

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
make check
make report
```

## Changing a feature or a model

1. Declare the feature's direction and its adverse-action wording in `features.py`.
2. Run `make report` and read sections 5 to 7 of the validation report: monotonicity, stability
   and fairness. Say in the pull request how they changed.
3. A change that makes a policy check pass by loosening `policy/model_risk.yaml` needs its own
   justification in the pull request.
