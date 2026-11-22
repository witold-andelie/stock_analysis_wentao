# Development rules for humans and coding agents

This project is developed **model-first**. The OPM model in
[`docs/architecture/opm/`](docs/architecture/opm/OPL.md) (Graphviz DOT + OPL text) is the
architecture contract.

1. **Start from the model.** Before writing code for a new feature, add or change the process,
   object or link in the relevant `SD*.dot` file and its OPL sentence. Mark new items ⬜ PLANNED
   (dotted, `fillcolor="#f2f2f2"`) until they are implemented.
2. **One commit = diagram + OPL + code + test.** When a planned item lands, make it solid,
   switch its OPL status to ✅, and add or extend a test in `tests/`. If a diagram, `OPL.md` and the
   code disagree, fix all three together.
3. **Respect the owner boundaries** (fill colours in `legend.dot`): ingestion modules never compute
   features; serving modules (`api`, `mcp_server`, `service`, `dashboard`) read **only** the
   Artifact Store (`driftrev.artifacts`), never the DuckDB warehouse.
4. **Keep the invariants written in OPL.** Specifically: no look-ahead (row *t* uses data available
   by the close of *t*), TEST is never used to choose anything, all tunables live in `config.py`,
   order submission stays behind `confirm=true` + `DRIFTREV_ALLOW_ORDERS=true` and paper by default,
   and data never goes to GitHub.
5. **Render and test before committing:**
   ```bash
   .venv/bin/python scripts/render_opm.py
   PYTHONPATH=src .venv/bin/python -m pytest tests -q
   ```
6. **The roadmap lives in `SD6_roadmap.dot`** (R1…R7). Pick work from there; extend it there first. Scope is US equities
   and German equities on Deutsche Börse Xetra, DAX + MDAX + SDAX (R7; select with `DRIFTREV_MARKET=us|de`).
   China A-shares are explicitly out of scope.
