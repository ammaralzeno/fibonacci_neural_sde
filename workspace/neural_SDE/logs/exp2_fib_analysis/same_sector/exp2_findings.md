# Exp 2 findings — 00674201, 14335601, 13376801, 01272601, 00764701

Aligned 2015-12-30 .. 2020-12-30, 1260 windows; near-level = |D| < 0.02.

(a) Levels meaningful per asset: near-level rate 28%-33%; FSM events on every asset (Bounce 15-22, Hover 29-33 each).
(b) Simultaneity: co-occurrence lift 1.25 (1.0 = independence); lead/lag peaks at lag 0 (|corr| = 0.113).
(c) Cross-asset effects, top KS pairs (A near level -> B's next-day returns):
    00764701 -> 01272601: KS = 0.09 (p = 0.014), mean|ret| 0.0177 vs 0.0158 unconditional
    01272601 -> 00764701: KS = 0.08 (p = 0.050), mean|ret| 0.0163 vs 0.0133 unconditional

Figures: `figA_simultaneous_levels.png`; combined: `../exp2_basket_comparison.png`.
