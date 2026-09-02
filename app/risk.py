def position_size(capital: float, risk_pct: float, entry: float, stop: float) -> int:
    if capital <= 0:
        raise ValueError("capital must be > 0")
    if not (0 < risk_pct < 1):
        raise ValueError("risk_pct must be a decimal between 0 and 1, e.g. 0.005")
    per_share_risk = abs(entry - stop)
    if per_share_risk <= 0:
        raise ValueError("entry and stop must differ")
    max_risk = capital * risk_pct
    return int(max_risk // per_share_risk)
