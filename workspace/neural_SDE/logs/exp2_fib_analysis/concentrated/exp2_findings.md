# Experiment 2 — Multi-asset Fibonacci features: findings

Basket (4 assets): 00177301, 00187801, 00169001, 00116101
Aligned calendar: 2015-12-30 .. 2017-12-28; rolling 252d windows: 504 window-ends.
Near-level definition: |D| < 0.02 (paper's +/-0.02 band, normalized by window range).

## (a) Are Fibonacci levels meaningful independently for each asset?

- Near-level rate per asset: 00177301: 28.0%, 00187801: 31.0%, 00169001: 33.3%, 00116101: 26.4%
- FSM interaction events on real data (Bounce/Break/Hover/Timeout per asset):
  - 00177301: Bounce=12, Break=3, Hover=17, Timeout=9
  - 00187801: Bounce=5, Break=4, Hover=16, Timeout=7
  - 00169001: Bounce=11, Break=4, Hover=13, Timeout=7
  - 00116101: Bounce=16, Break=1, Hover=12, Timeout=3

Figures: `figA_simultaneous_levels.png`, `figB_distance_heatmap.png`, `check_a_signed_distance_hist.png`, `check_a_fsm_events.png`.

## (b) Do different assets reach Fibonacci levels simultaneously?

- Mean pairwise co-occurrence lift: 1.01 (1.0 = independence; >1 = simultaneous more often than chance).
- Max lift: 1.21; min lift: 0.86.
- Lead/lag: mean pairwise indicator cross-correlation peaks at lag -7 days (|corr|=0.039).

Figures: `check_b_cooccurrence_lift.png`, `check_b_lagged_xcorr.png`.

## (c) Do Fibonacci events in one asset coincide with movements in another?

Ordered pairs (A near level -> B's next-day returns), top KS statistic pairs:
- 00177301 -> 00169001: KS=0.17 (p=0.006), n=141, mean|ret| 0.0099 vs 0.0086 unconditional
- 00187801 -> 00116101: KS=0.14 (p=0.030), n=156, mean|ret| 0.0339 vs 0.0263 unconditional
- 00177301 -> 00187801: KS=0.13 (p=0.058), n=141, mean|ret| 0.0169 vs 0.0122 unconditional
- 00116101 -> 00187801: KS=0.12 (p=0.131), n=133, mean|ret| 0.0128 vs 0.0138 unconditional

Figures: `check_c_conditional_ks_heatmap.png`, `check_c_conditional_overlays.png`.
