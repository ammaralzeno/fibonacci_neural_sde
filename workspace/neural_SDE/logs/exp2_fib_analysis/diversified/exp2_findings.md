# Experiment 2 — Multi-asset Fibonacci features: findings

Basket (4 assets): 00138001, 00141402, 00107801, 00116101
Aligned calendar: 2015-12-30 .. 2017-12-28; rolling 252d windows: 504 window-ends.
Near-level definition: |D| < 0.02 (paper's +/-0.02 band, normalized by window range).

## (a) Are Fibonacci levels meaningful independently for each asset?

- Near-level rate per asset: 00138001: 24.6%, 00141402: 29.2%, 00107801: 31.5%, 00116101: 26.4%
- FSM interaction events on real data (Bounce/Break/Hover/Timeout per asset):
  - 00138001: Bounce=10, Break=4, Hover=14, Timeout=5
  - 00141402: Bounce=9, Break=2, Hover=10, Timeout=5
  - 00107801: Bounce=8, Break=6, Hover=13, Timeout=6
  - 00116101: Bounce=16, Break=1, Hover=12, Timeout=3

Figures: `figA_simultaneous_levels.png`, `figB_distance_heatmap.png`, `check_a_signed_distance_hist.png`, `check_a_fsm_events.png`.

## (b) Do different assets reach Fibonacci levels simultaneously?

- Mean pairwise co-occurrence lift: 1.03 (1.0 = independence; >1 = simultaneous more often than chance).
- Max lift: 1.22; min lift: 0.86.
- Lead/lag: mean pairwise indicator cross-correlation peaks at lag -7 days (|corr|=0.017).

Figures: `check_b_cooccurrence_lift.png`, `check_b_lagged_xcorr.png`.

## (c) Do Fibonacci events in one asset coincide with movements in another?

Ordered pairs (A near level -> B's next-day returns), top KS statistic pairs:
- 00138001 -> 00107801: KS=0.10 (p=0.255), n=124, mean|ret| 0.0084 vs 0.0086 unconditional
- 00138001 -> 00141402: KS=0.10 (p=0.315), n=124, mean|ret| 0.0127 vs 0.0122 unconditional
- 00107801 -> 00138001: KS=0.09 (p=0.359), n=159, mean|ret| 0.0175 vs 0.0196 unconditional
- 00116101 -> 00141402: KS=0.09 (p=0.448), n=133, mean|ret| 0.0114 vs 0.0127 unconditional

Figures: `check_c_conditional_ks_heatmap.png`, `check_c_conditional_overlays.png`.
