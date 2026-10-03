# LLM-DR

Research materials for persona-grounded agents in a shared on-demand delivery simulation. This release contains the executable environment, empirical inputs with pseudonymized identifiers, four rule-based reference policies, LLM-DR reference outputs, evaluation scripts, and the current manuscript.

The released task is **conditional single-day reconstruction**: rider personas and an observed order stream condition repeated order actions, shared assignment, and routed execution. Evaluation combines delivery performance, empirical behavioral similarity, and collective temporal and spatial outcomes. The environment and evaluation protocol can support comparisons between agent policies; LLM-DR provides one reference implementation.

## Quick Start

Use Python 3.11 or newer. The release checks were run with Python 3.11.15.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/reproduce.py verify
python scripts/reproduce.py evaluate
python scripts/reproduce.py smoke
```

These commands make no LLM or routing API requests. `verify` checks the release files and frozen action records. `evaluate` recomputes behavioral discrepancies from the released empirical data and frozen simulation outputs. `smoke` runs a small rule-based, straight-line-routing test; it is a software check, separate from the paper's experiment.

## Fresh Simulations

```bash
cp .env.example .env.local
python scripts/reproduce.py simulate
```

The command prints the fully specified main experiment without executing it. Add your credentials to `.env.local`, then explicitly opt in to API requests:

```bash
python scripts/reproduce.py simulate --execute --models deepseek_v4
python scripts/reproduce.py simulate --execute --models nearest_baseline highest_fee_baseline profit_baseline persona_heuristic
```

All main configurations use 60 riders, 15 per persona, 08:00-20:00 input, Amap bicycling routes, strict routing, and **no global or per-step rider-decision cap**. The wrapper overrides the original runner's small-scale defaults. Fresh LLM runs incur provider charges and need not reproduce historical decisions exactly. Order identifiers have been consistently pseudonymized, so their numeric labels in prompts differ from the original run.

## Contents

| Directory | Contents |
| --- | --- |
| `data/empirical/` | Order inputs, filtered persona orders, and wave action records |
| `data/persona/` | Persona assignments, feature tables, summaries, and clustering diagnostics |
| `config/` | Main, smoke, and model connection configurations |
| `reference/simulation/` | Frozen LLM-DR and rule-policy outputs used for result verification |
| `scripts/` | Simulation, persona extraction, evaluation, release checks, and plotting |
| `paper/` | LaTeX source, figure assets, fonts and notices, and compiled manuscript |
| `docs/` | Task protocol, data documentation, and agent extension points |

Read [the task protocol](docs/PROTOCOL.md) before comparing methods and [the data card](docs/DATA_CARD.md) before interpreting sample sizes or redistributing data.

See [release verification](docs/RELEASE_CHECKS.md) for the checks completed on this copy, including recovery of the legacy filtered file's tracking identifiers.

## Persona Extraction and Figures

```bash
python scripts/reproduce.py personas
python scripts/reproduce.py figures
```

Persona extraction repeats the documented clustering settings and writes to `outputs/persona`. Plotting regenerates the distribution and simulated action-chain views from released records. Geographic plotting may request CARTO map tiles; some manuscript illustrations and manually edited layouts are distributed as static assets in `paper/`. Frozen route geometry is retained in simulation wave records; the original live routing cache is not distributed.

To compile the manuscript:

```bash
cd paper
latexmk -xelatex -interaction=nonstopmode -halt-on-error main.tex
```

## Extending Agents

New OpenAI-compatible models can be configured in `config/model_suite.example.json`. Rule-based clients can be added at the extension points documented in [AGENT_INTERFACE.md](docs/AGENT_INTERFACE.md). Keep observation, assignment, routing, and scoring conventions fixed for comparisons on the same task. Record any changes as a separate task configuration.

## Release Status

Project code and documentation are licensed under [MIT](LICENSE); authorized project data and derived tabular results are licensed under [CC BY 4.0](data/LICENSE.md). The manuscript, model-return text, route geometry, and third-party assets are handled separately in [LICENSE_NOTICE.md](LICENSE_NOTICE.md).

The materials are maintained in the **private repository** [Nicholas0027/LLM-DR](https://github.com/Nicholas0027/LLM-DR), pending final public-release review. Pseudonymization retains detailed geography and time, so it does not guarantee that trajectories are unidentifiable. API credentials, old notebooks, private identifier crosswalks, map-tile caches, and historical development outputs are excluded.

The current manuscript remains anonymous. GitHub account identity, citation metadata, and any public paper release should be coordinated before making the repository public.
