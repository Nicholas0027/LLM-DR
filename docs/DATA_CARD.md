# Data Card

## Release Scope

The owner has authorized redistribution of de-identified delivery records. This staged release substitutes consistent numeric aliases for source courier and order identifiers. Original identifier crosswalks remain outside the repository. Shop IDs, area-of-interest IDs, unused weather fields, raw routing-service responses, and unused source row indices are omitted.

Geographic coordinates and time resolution are retained to support routing and behavioral verification. Identifier removal alone does not provide a formal anonymity guarantee: trajectories, destinations, and schedules may support linkage. Public release requires a final privacy and third-party-rights review. As an ethical research practice, avoid attempts to identify riders or customers; this request does not modify the CC BY 4.0 license terms.

## Files and Sample Sizes

| File | Records | Use |
| --- | ---: | --- |
| `data/empirical/orders.csv` | 9,199 | Original simulation input, with identifier aliases and column minimization |
| `data/empirical/orders_filtered.tsv` | 9,025 | Filtered order input used for persona feature extraction; UTF-16 TSV |
| `data/empirical/waves.csv` | 2,508 | Routed wave action records and aggregate distances/durations |
| `data/persona/rider_persona_assignments.csv` | 710 | Retained empirical rider profiles and persona labels |

The simulation selects 8,656 orders by service date and 08:00-20:00 creation time from `orders.csv`. Persona extraction uses the 9,025-row filtered subset; empirical distribution evaluation uses the original order file and wave sample with the retained persona cohort. The complete input file and the filtered subset have different row counts and roles.

The legacy filtered export had collapsed its tracking identifiers into one floating-point value. In this release copy, every filtered record was matched uniquely to the original order stream using rider ID, creation time, and pickup/drop-off coordinates; all 9,025 records matched exactly, with no unmatched or ambiguous records. Original tracking IDs were recovered from those matches before applying the release pseudonyms. The original workspace file is untouched. Persona features do not use tracking IDs, so this repair preserves the clustering inputs and reproduced assignments. `release_provenance.json` records the repair.

Wave records contain 711 distinct courier identifiers before the feature-retention step; 710 rider profiles enter the final persona extraction. The 08:00-20:00 comparison includes 705 retained empirical riders: 281 Full-time Workhorse, 111 Lunch-Peak Specialist, 219 Super Stacker, and 94 Dinner-Peak Core.

## Coordinates and Time

Empirical order coordinates retain the source's obfuscated frame. Existing loaders apply the documented constant offset to obtain the simulation frame. Simulation endpoints and Amap route polylines retain their original frames; plotting code performs the required conversions for basemap alignment. Do not apply these transformations twice.

Order/event timestamps are Unix seconds in empirical files; loaders interpret them in Asia/Shanghai. Simulation logs use timezone-aware ISO timestamps. Preserve event ordering and distinguish creation, expected action times, and simulated wave-end completion.

## Wave Representation

`tracking_ids_chronological` lists order IDs for recorded pickup/drop-off events. `action_types_chronological` and `expect_times_chronological` each start with one ASSIGN event; the ID list starts at the following pickup event and therefore has one fewer entry. The feature loader removes the initial ASSIGN action before pairing actions with IDs. `expect_times_chronological` retains the source column name and supplies the event-time sequence used in the current analysis; the owner confirms the action ordering comes from execution records. `real_distances_segments` and aggregate navigation fields support the existing feature pipeline.

This release retains the reconstructed routed-wave sample rather than claiming exhaustive coverage of every service episode. All 2,508 supplied waves contain at least two distinct orders. Wave-specific indicators describe this sample; order-level indicators use the separate order records. Continuous empirical GPS traces and raw routing-response payloads are not part of this release.

## Frozen Simulation Records

Five policy folders contain rider summaries, completed-order logs, wave records, decisions, and runtime metadata. Empirical couriers and synthetic rider IDs are separate populations. The latter's IDs 1-60 identify simulated agents, not actual workers.

Source order IDs are replaced consistently in JSON fields, returned model text, CSVs, and the manuscript release copy. Other model rationale wording is retained, including any numerical judgment errors. Rationale text is a reported model output; executed actions and assignments are established by structured logs.

## Licensing and Attribution

Project code uses MIT. The authorized empirical data, persona tables, and project-derived tabular results use CC BY 4.0; see `data/LICENSE.md` for attribution and scope. Permission to release project data does not establish a redistribution license for third-party map tiles, fonts, routing-service payloads, or included LaTeX packages. Map-tile caches and the original live routing cache are excluded. Fonts and LaTeX assets retain their existing notices. `LICENSE_NOTICE.md` records the separately reviewed materials, and source publications are listed in `paper/references.bib`.
