"""
Trading plan model.

Pipeline -> TradePlan -> Risk -> Paper Broker
"""

from dataclasses import dataclass
from datetime import datetime


@dataclass(slots=True)
class TradePlan:

    ticker: str

    asset_type: str

    action: str

    entry_price: float

    stop_loss: float

    take_profit: float

    confidence: int

    overall_score: int

    technical_score: int

    fundamental_score: int

    news_score: int

    risk_reward: float

    max_risk_percent: float

    created_at: str

    strategy: str = "Radar V1"

    notes: str = ""

    @property
    def stop_distance(self) -> float:
        return abs(self.entry_price - self.stop_loss)

    @property
    def target_distance(self) -> float:
        return abs(self.take_profit - self.entry_price)

    @classmethod
    def create(
        cls,
        *,
        ticker: str,
        asset_type: str,
        action: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        confidence: int,
        overall_score: int,
        technical_score: int,
        fundamental_score: int,
        news_score: int,
        max_risk_percent: float = 1.0,
        strategy: str = "Radar V1",
        notes: str = "",
    ):

        risk = abs(entry_price - stop_loss)

        reward = abs(take_profit - entry_price)

        if risk == 0:
            rr = 0.0
        else:
            rr = round(reward / risk, 2)

        return cls(
            ticker=ticker,
            asset_type=asset_type,
            action=action,
            entry_price=round(entry_price, 2),
            stop_loss=round(stop_loss, 2),
            take_profit=round(take_profit, 2),
            confidence=confidence,
            overall_score=overall_score,
            technical_score=technical_score,
            fundamental_score=fundamental_score,
            news_score=news_score,
            risk_reward=rr,
            max_risk_percent=max_risk_percent,
            created_at=datetime.utcnow().isoformat(),
            strategy=strategy,
            notes=notes,
        )

    def print_summary(self):

        print("=" * 70)

        print(f"Ticker        : {self.ticker}")

        print(f"Asset         : {self.asset_type}")

        print(f"Action        : {self.action}")

        print(f"Entry         : {self.entry_price}")

        print(f"Stop Loss     : {self.stop_loss}")

        print(f"Take Profit   : {self.take_profit}")

        print(f"Risk Reward   : {self.risk_reward}")

        print(f"Confidence    : {self.confidence}%")

        print(f"Overall Score : {self.overall_score}")

        print(f"Technical     : {self.technical_score}")

        print(f"Fundamental   : {self.fundamental_score}")

        print(f"News          : {self.news_score}")

        print("=" * 70)