# AI-Stock-Radar Codex Working Rules

These rules are binding for Codex work in this repository.

## Project and pathway

- AI-Stock-Radar is a deterministic Python trading platform.
- AI is not the primary trading or execution brain. Trading decisions, portfolio state transitions, risk controls, and execution behavior must remain deterministic and auditable.
- The approved development pathway must not be reordered.
- Paper trading must not be moved forward ahead of the approved research and validation pathway.
- Real-money execution must not be enabled unless the user explicitly authorizes that stage.

## Protected behavior

- `src/main.py` is protected and must not be modified unless explicitly requested.
- Existing strategy rules, risk parameters, execution semantics, fee assumptions, slippage assumptions, and portfolio event ordering must not change unless explicitly requested.
- Next-available-Open execution and other established causal behavior must not be changed implicitly.
- Gap stops execute at the actual available Open.
- Deterministic candidate tie-breaking must be preserved.
- Do not introduce ticker-specific parameters without explicit approval.
- Do not silently use alternate, legacy, backup, or paper-path defaults as the active research baseline.
- Attribution and diagnostic tasks must remain observational unless an implementation task explicitly authorizes strategy changes.
- Do not access or modify `.env`, credentials, secrets, API keys, or broker credentials.
- Do not delete, replace, rewrite, or move existing historical research artifacts or backup files.

## Approval workflow

Before any future implementation:

1. Inspect the relevant code and tests.
2. Describe the intended change and behavioral scope.
3. List every file planned for creation or modification.
4. Wait for explicit approval before editing.

After an approved implementation, report:

- every changed file;
- every test or verification command run;
- the result of each command;
- assumptions made;
- behavioral differences introduced, including confirmation when there are none.

Do not optimize parameters, introduce ticker-specific behavior, advance the roadmap, or expand execution authority without explicit approval.
