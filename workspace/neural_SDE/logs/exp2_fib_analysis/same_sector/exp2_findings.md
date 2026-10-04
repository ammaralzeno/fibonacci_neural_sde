# Experiment 2 — Multi-asset Fibonacci features: findings

Basket (5 assets): 00674201, 14335601, 13376801, 01272601, 00764701
Aligned calendar: 2015-12-30 .. 2017-12-28; rolling 252d windows: 504 window-ends.
Near-level definition: |D| < 0.02 (paper's +/-0.02 band, normalized by window range).

## (a) Are Fibonacci levels meaningful independently for each asset?

- Near-level rate per asset: 00674201: 35.5%, 14335601: 29.4%, 13376801: 26.4%, 01272601: 36.5%, 00764701: 32.9%
- FSM interaction events on real data (Bounce/Break/Hover/Timeout per asset):
  - 00674201: Bounce=6, Break=5, Hover=15, Timeout=7
  - 14335601: Bounce=12, Break=2, Hover=16, Timeout=10
  - 13376801: Bounce=19, Break=1, Hover=13, Timeout=5
  - 01272601: Bounce=5, Break=1, Hover=13, Timeout=11
  - 00764701: Bounce=10, Break=3, Hover=14, Timeout=8

Figures: `figA_simultaneous_levels.png`, `figB_distance_heatmap.png`, `check_a_signed_distance_hist.png`, `check_a_fsm_events.png`.

## (b) Do different assets reach Fibonacci levels simultaneously?

- Mean pairwise co-occurrence lift: 1.36 (1.0 = independence; >1 = simultaneous more often than chance).
- Max lift: 1.67; min lift: 0.96.
- Lead/lag: mean pairwise indicator cross-correlation peaks at lag 0 days (|corr|=0.176).

Figures: `check_b_cooccurrence_lift.png`, `check_b_lagged_xcorr.png`.

## (c) Do Fibonacci events in one asset coincide with movements in another?

Ordered pairs (A near level -> B's next-day returns), top KS statistic pairs:
- 13376801 -> 00764701: KS=0.15 (p=0.027), n=133, mean|ret| 0.0131 vs 0.0121 unconditional
- 00764701 -> 01272601: KS=0.13 (p=0.039), n=166, mean|ret| 0.0116 vs 0.0096 unconditional
- 13376801 -> 14335601: KS=0.13 (p=0.072), n=133, mean|ret| 0.0110 vs 0.0102 unconditional
- 00764701 -> 00674201: KS=0.11 (p=0.121), n=166, mean|ret| 0.0156 vs 0.0119 unconditional

Figures: `check_c_conditional_ks_heatmap.png`, `check_c_conditional_overlays.png`.
