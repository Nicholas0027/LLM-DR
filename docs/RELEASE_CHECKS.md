# Release Verification

Checked on 2026-10-02 using Python 3.11.15, NumPy 2.4.3, pandas 2.3.3, and TeX Live 2026.

| Check | Result |
| --- | --- |
| Credential and original rider/order-ID scan of release text files | Passed |
| Legacy filtered-order ID recovery | All 9,025 records matched one-to-one; no missing or ambiguous matches |
| Restored filtered records against original stream | All released order columns agree with the corresponding original records |
| Feature extraction from release inputs | Matches frozen features within numerical tolerance |
| Full four-persona extraction, including clustering diagnostics | All 710 rider labels reproduced exactly |
| Five frozen 60-rider runs | Decision, completion, rider, and wave totals are internally consistent |
| Offline empirical evaluation | All five reference validation tables reproduced within numerical tolerance |
| Small rule-policy smoke run | Passed; 8 riders, 8 decisions, no LLM or routing requests |
| Main-command preview | 60 riders, strict Amap bicycling, both decision caps set to zero; no execution |
| Manuscript release-copy compilation | 17 pages; no undefined references or overfull boxes |
| Release-copy appendix rendering and extracted-text ID scan | Passed |

Feature-table checks allow `rtol=1e-7, atol=1e-7`; recomputed evaluation uses `rtol=1e-8, atol=1e-8`. The largest observed feature difference after CSV reserialization was approximately 6e-8 square kilometers in activity area, with all persona labels unchanged.

The original empirical datasets and research scripts are untouched. The manuscript release copy follows the current `1002version`, including the owner's requested benchmark-framing revision. It pseudonymizes identifiers and records that transformation in the appendix. The data copy restores the filtered file's IDs by exact record linkage, and the repository adds wrappers/documentation. The seven research scripts match their source SHA-256 hashes in `release_provenance.json`.

Fresh paid LLM/Amap runs and geographic figure regeneration were not performed during release preparation. Frozen geometry and existing manuscript figure assets are supplied. Check `LICENSE_NOTICE.md` and `DATA_CARD.md` before making this repository public. The authenticated repository address was verified on 2026-10-03: [Nicholas0027/LLM-DR](https://github.com/Nicholas0027/LLM-DR). The repository remains private pending public-release review.
