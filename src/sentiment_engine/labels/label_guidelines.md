# Label Guidelines

These labels are for research and shadow-mode evaluation. They are not trading advice.

## Sentiment / Market Implication

- `bullish_market`: the text plausibly supports risk-on NASDAQ conditions, easier financial conditions, lower policy uncertainty, or explicit equity/technology strength.
- `bearish_market`: the text plausibly increases trade, fiscal, regulatory, or policy uncertainty for NASDAQ-sensitive assets.
- `geopolitical_risk`: the text plausibly increases geopolitical, energy, safe-haven, or tail-risk concerns.
- `neutral`: no clear short-horizon NASDAQ implication.
- `volatility_only`: the text may increase realised range but direction is unclear.
- `low_confidence`: insufficient evidence to label.

## Tradeability

- `tradeable_directional`: directional label is clear enough for research evaluation and no whipsaw condition dominates.
- `volatility_only`: elevated movement risk, weak direction.
- `no_trade_whipsaw`: contradiction, burst, or market-path reversal risk dominates.
- `no_impact`: no actionable market hypothesis.
- `ambiguous`: label requires human review.

## Human Review Rules

1. Preserve original text and media flags.
2. Do not use post-event price movement to assign live-time sentiment labels.
3. Use market-path labels only for research targets and detector evaluation.
4. Mark scheduled macro-news proximity separately; do not credit the post for moves during blackout windows without a matched-control test.
5. Record uncertainty rather than forcing a directional label.
