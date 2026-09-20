# Experiment 2 — Multi-asset Fibonacci features: findings

Basket (5 assets): 02531301, 02531801, 10024392W, 10876801W, 16032902W
Aligned calendar: 2016-01-19 .. 2017-12-28; rolling 252d windows: 475 window-ends.
Near-level definition: |D| < 0.02 (paper's +/-0.02 band, normalized by window range).

## (a) Are Fibonacci levels meaningful independently for each asset?

- Near-level rate per asset: 02531301: 30.7%, 02531801: 30.7%, 10024392W: 30.1%, 10876801W: 22.3%, 16032902W: 25.5%
- FSM interaction events on real data (Bounce/Break/Hover/Timeout per asset):
  - 02531301: Bounce=10, Break=1, Hover=12, Timeout=9
  - 02531801: Bounce=13, Break=6, Hover=20, Timeout=8
  - 10024392W: Bounce=15, Break=7, Hover=13, Timeout=2
  - 10876801W: Bounce=9, Break=3, Hover=15, Timeout=7
  - 16032902W: Bounce=13, Break=4, Hover=14, Timeout=6

Figures: `figA_simultaneous_levels.png`, `figB_distance_heatmap.png`, `check_a_signed_distance_hist.png`, `check_a_fsm_events.png`.

## (b) Do different assets reach Fibonacci levels simultaneously?

- Mean pairwise co-occurrence lift: 0.96 (1.0 = independence; >1 = simultaneous more often than chance).
- Max lift: 1.24; min lift: 0.71.
- Lead/lag: mean pairwise indicator cross-correlation peaks at lag 0 days (|corr|=0.017).

Figures: `check_b_cooccurrence_lift.png`, `check_b_lagged_xcorr.png`.

## (c) Do Fibonacci events in one asset coincide with movements in another?

Ordered pairs (A near level -> B's next-day returns), top KS statistic pairs:
- 10876801W -> 10024392W: KS=0.15 (p=0.043), n=106, mean|ret| 0.0110 vs 0.0083 unconditional
- 02531301 -> 02531801: KS=0.15 (p=0.025), n=146, mean|ret| 0.0106 vs 0.0124 unconditional
- 02531801 -> 16032902W: KS=0.13 (p=0.065), n=146, mean|ret| 0.0114 vs 0.0093 unconditional
- 10024392W -> 02531301: KS=0.12 (p=0.094), n=143, mean|ret| 0.0086 vs 0.0082 unconditional

Figures: `check_c_conditional_ks_heatmap.png`, `check_c_conditional_overlays.png`.
