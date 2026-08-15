# Pushdown optimization checklist

Primary submode: `exploit`—measure and remove costs in the current faithful Pushdown implementation before exploring new attention formulations.

- [x] Keep one fixed incumbent benchmark contract.
- [x] Define controls that isolate manual depth-gradient, full bias, and attachment costs.
- [x] Rank candidates by expected end-to-end gain, semantic risk, and implementation cost.
- [x] Promote only measured lead changes after baseline attribution.
- [x] Require reference oracle and depth-gradient parity.
- [x] Re-run the incumbent and legacy control after diagnostic switches were added.
- [x] Record failed or deferred candidates rather than silently dropping them.
