"""Shared data models for the AI-Stock-Radar pipeline."""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FastScanResult:
    """Stage 1 technical fast-scan result."""

    ticker: str
    asset_type: str

    technical_score: int
    technical_stars: int

    technical_signal: str
    technical_confidence: int

    close: float
    ema20: float
    ema50: float
    rsi14: float
    macd: float

    reasons: list[str] = field(default_factory=list)

    @property
    def is_stock(self) -> bool:
        """Return whether this result belongs to a stock."""

        return self.asset_type == "stock"

    @property
    def is_crypto(self) -> bool:
        """Return whether this result belongs to cryptocurrency."""

        return self.asset_type == "crypto"


@dataclass(frozen=True)
class DeepResult:
    """Stage 2 deep-analysis result."""

    ticker: str
    asset_type: str

    overall_score: int
    technical_score: int
    fundamental_score: int
    news_score: int

    signal: str
    confidence: int
    stars: int

    technical_signal: str
    news_sentiment: str

    technical_reasons: list[str] = field(default_factory=list)
    fundamental_reasons: list[str] = field(default_factory=list)
    news_headlines: list[str] = field(default_factory=list)
    decision_reasons: list[str] = field(default_factory=list)

    @property
    def is_stock(self) -> bool:
        """Return whether this result belongs to a stock."""

        return self.asset_type == "stock"

    @property
    def is_crypto(self) -> bool:
        """Return whether this result belongs to cryptocurrency."""

        return self.asset_type == "crypto"


@dataclass(frozen=True)
class CandidateSelection:
    """Final Python-selected candidates before AI analysis."""

    top_stocks: list[DeepResult]
    top_crypto: list[DeepResult]
    selected_crypto: DeepResult | None

    @property
    def ai_candidates(self) -> list[DeepResult]:
        """Return three stocks and one selected crypto."""

        candidates = list(self.top_stocks[:3])

        if self.selected_crypto is not None:
            candidates.append(self.selected_crypto)

        return candidates


@dataclass(frozen=True)
class PipelineSummary:
    """Summary statistics for a complete pipeline run."""

    total_stocks: int
    total_crypto: int

    successful_fast_stocks: int
    successful_fast_crypto: int

    deep_stock_count: int
    deep_crypto_count: int

    final_stock_count: int
    final_crypto_count: int

    failed_symbols: list[str] = field(default_factory=list)

    @property
    def total_symbols(self) -> int:
        """Return the configured total number of assets."""

        return self.total_stocks + self.total_crypto

    @property
    def successful_fast_total(self) -> int:
        """Return the total successful Stage 1 scans."""

        return (
            self.successful_fast_stocks
            + self.successful_fast_crypto
        )