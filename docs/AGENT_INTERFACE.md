# Agent Extension Points

The release preserves the research implementation. It does not introduce a second environment or a new dispatch algorithm.

## OpenAI-Compatible Models

Add a model entry to `config/model_suite.example.json` with `base_url`, `model`, `api_key_env`, and inference settings. Credentials are loaded from environment variables; keep `.env.local` untracked. Invoke the main wrapper with `--models YOUR_MODEL --execute` only when you intend to make paid requests.

The model receives the prompt built by `build_prompt` in `run_simulation_experiment.py`. The action format is a JSON object with `accepted_order_ids` and a visible rationale. Accepted IDs must belong to the current offered set; an empty list rejects the batch. The environment validates the proposal, resolves exclusive assignments, constructs the fixed waypoint sequence, invokes routing, and advances state.

## Rule or Learned Policies

`RuleClient.decide_for_rider` in `run_scaled_simulation_experiment.py` receives rider state, offered orders, and simulation time and returns a `DecisionResult`. `client_for_model` and `decide_with_client` select and invoke the client. These are the minimal integration points for another policy; preserve the shared dispatch and execution functions.

For protocol-compliant comparisons, new policies must use the local observation specified in the paper and must not inspect future arrivals, empirical action labels, or hidden global order state. The current research runner is trusted-code execution; it is not a sandbox that enforces this information boundary against arbitrary uploaded code. Document the fields and history available to each policy.

## Reporting a New Run

Record the task configuration, agent/model version, prompt changes, initial-state seed, routing provider, inference settings, observed information, and policy-specific runtime cost. Report both fleet and persona denominators consistently. Changed dispatch, availability, route insertion, reward, or completion rules define a different task variant and should not be silently pooled with the frozen reference results.
