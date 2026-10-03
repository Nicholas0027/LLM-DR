# Single-Day Reconstructive Task

## Task and Reference Implementation

A rider chooses a subset of locally offered order IDs; the empty subset denotes rejection. The shared environment resolves competing choices, constructs routed pickup/drop-off waves, and updates each rider's state. LLM-DR is the persona-grounded LLM reference policy. Other references are nearest order, highest fee, fee-distance optimization, and persona heuristic.

The empirical persona profiles are input conditions. Distributional evaluation examines how those conditions carry through to executed behavior. The task is a conditional reconstruction rather than a held-out prediction protocol; additional days or cities should be released as separately specified tasks.

## Fixed Main Configuration

| Setting | Value |
| --- | --- |
| Service date and input window | 2020-02-15, 08:00-20:00, Asia/Shanghai |
| Orders exposed during the window | 8,656 |
| Fleet | 60 riders, 15 per persona |
| Initial placement | Demand-density sampling, seed 20260625, 0.45-km jitter |
| Simulation step | 10 minutes |
| Pending horizon | 30 minutes |
| Local radius | 2 km |
| Candidate riders per order | At most 3 |
| Offers per rider | At most 6 |
| Global/per-step decision cap | Unlimited (`0`) |
| Availability | Idle riders can receive offers and choose orders |
| Memory in the LLM prompt | Up to 5 recently completed orders |
| Assignment | Distance-prioritized conflict resolution |
| Wave sequence | All assigned pickups, then corresponding drop-offs |
| Routing | Amap bicycling, strict mode |
| Service allowance | 120 seconds per order |
| Order completion | Common wave-end timestamp |
| Rewards | Assumed CNY 3/6/10/15 distance brackets |

Orders created within a 10-minute batch are exposed at its start. Work already assigned is drained after the input window. LLM requests use temperature 0.2, up to 1,024 output tokens, structured accepted IDs, and the documented retry/invalid-action handling. Planning in this implementation selects offered orders and bundle composition; waypoint construction and execution are environmental mechanisms.

## Evaluation

Delivery indicators include orders, rewards, mileage, bundle size, stacked-wave share, and the wave-end on-time proxy. Rider-level reports should state their denominator and persona composition.

Empirical similarity includes creation-time JSD, pickup-grid JSD, OD and routed-wave Wasserstein-1/KS distances, pooled pickup radius-of-gyration error, and pooled pickup-area log error. Lower discrepancy indicates closer agreement with the specified empirical reference.

Rank-percentile plots compare relative positions within each source population. They use 705 empirical riders in the evaluation window and 60 simulated riders, with all four personas ranked together within each source. Their percentiles complement untransformed discrepancies. Wave indicators use the routed wave sample, whose inclusion process is described in the data card.

Fleet analysis describes concurrent service, wave-end completions, daily activity-space hulls, and executed road use. Comparisons should preserve the current aggregation and timestamp conventions and report input coverage alongside rider-level outcomes.

## Reproduction Modes

1. **Frozen-record verification:** audit included decisions, assignments, waves, and aggregates; no external APIs.
2. **Offline empirical evaluation:** recompute discrepancies from included empirical records and frozen simulation logs.
3. **Fresh simulation:** execute the configured policy and routing provider with new requests. Identifier replacement, provider/model changes, and stochastic responses can change the resulting decisions.

Frozen results and fresh runs should be identified separately in reports. The straight-line smoke test is only an installation and software check and must not be reported as the main road-network experiment.
