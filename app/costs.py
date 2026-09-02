from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class IntradayEquityCostModel:
    """Approximate Indian NSE equity-intraday charges.

    Defaults are deliberately configurable. Verify them against the latest Zerodha
    brokerage calculator before treating a production backtest as final.
    Percentages are expressed as decimal fractions (0.0003 == 0.03%).
    """

    brokerage_rate: float = 0.0003
    brokerage_cap_per_order: float = 20.0
    stt_sell_rate: float = 0.00025
    exchange_txn_rate: float = 0.0000307
    sebi_rate: float = 0.000001
    stamp_buy_rate: float = 0.00003
    gst_rate: float = 0.18

    def estimate(self, entry_price: float, exit_price: float, qty: int) -> dict[str, float]:
        if qty <= 0:
            return {"total": 0.0}

        buy_turnover = entry_price * qty
        sell_turnover = exit_price * qty
        turnover = buy_turnover + sell_turnover

        brokerage_buy = min(buy_turnover * self.brokerage_rate, self.brokerage_cap_per_order)
        brokerage_sell = min(sell_turnover * self.brokerage_rate, self.brokerage_cap_per_order)
        brokerage = brokerage_buy + brokerage_sell
        stt = sell_turnover * self.stt_sell_rate
        exchange = turnover * self.exchange_txn_rate
        sebi = turnover * self.sebi_rate
        stamp = buy_turnover * self.stamp_buy_rate
        gst = (brokerage + exchange + sebi) * self.gst_rate
        total = brokerage + stt + exchange + sebi + stamp + gst

        return {
            "brokerage": brokerage,
            "stt": stt,
            "exchange": exchange,
            "sebi": sebi,
            "stamp": stamp,
            "gst": gst,
            "total": total,
        }
