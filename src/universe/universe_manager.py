"""Universe configuration loader for AI-Stock-Radar."""

import json
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
UNIVERSE_DIRECTORY = PROJECT_ROOT / "config" / "universes"


@dataclass(frozen=True)
class Universe:
    """One configured market universe."""

    name: str
    region: str
    asset_type: str
    tickers: list[str]


@dataclass(frozen=True)
class MarketUniverse:
    """Combined stock and crypto universe."""

    stocks: list[str]
    crypto: list[str]

    @property
    def all_symbols(self) -> list[str]:
        """Return all configured symbols."""

        return self.stocks + self.crypto

    @property
    def stock_count(self) -> int:
        """Return the number of stocks."""

        return len(self.stocks)

    @property
    def crypto_count(self) -> int:
        """Return the number of crypto assets."""

        return len(self.crypto)

    @property
    def total_count(self) -> int:
        """Return the total number of configured assets."""

        return len(self.all_symbols)


def _load_json_file(file_name: str) -> Universe:
    """Load and validate one universe JSON file."""

    file_path = UNIVERSE_DIRECTORY / file_name

    if not file_path.exists():
        raise FileNotFoundError(
            f"Universe configuration not found: {file_path}"
        )

    with file_path.open(
        mode="r",
        encoding="utf-8",
    ) as file:
        payload = json.load(file)

    required_fields = {
        "name",
        "region",
        "asset_type",
        "tickers",
    }

    missing_fields = required_fields.difference(payload)

    if missing_fields:
        missing_text = ", ".join(sorted(missing_fields))
        raise ValueError(
            f"{file_name} is missing fields: {missing_text}"
        )

    tickers = payload["tickers"]

    if not isinstance(tickers, list):
        raise TypeError(
            f"`tickers` must be a list in {file_name}"
        )

    cleaned_tickers = []

    for ticker in tickers:
        if not isinstance(ticker, str):
            raise TypeError(
                f"Every ticker must be text in {file_name}"
            )

        normalized = ticker.strip().upper()

        if normalized and normalized not in cleaned_tickers:
            cleaned_tickers.append(normalized)

    if not cleaned_tickers:
        raise ValueError(
            f"No valid tickers found in {file_name}"
        )

    return Universe(
        name=str(payload["name"]),
        region=str(payload["region"]),
        asset_type=str(payload["asset_type"]),
        tickers=cleaned_tickers,
    )


def load_market_universe() -> MarketUniverse:
    """Load all configured stock and crypto universes."""

    us_stocks = _load_json_file(
        "stocks_us.json"
    )

    global_stocks = _load_json_file(
        "stocks_global.json"
    )

    crypto = _load_json_file(
        "crypto.json"
    )

    stocks = []

    for ticker in us_stocks.tickers + global_stocks.tickers:
        if ticker not in stocks:
            stocks.append(ticker)

    return MarketUniverse(
        stocks=stocks,
        crypto=crypto.tickers,
    )


def print_universe_summary(
    universe: MarketUniverse,
) -> None:
    """Print a short universe summary."""

    print("=" * 60)
    print("AI STOCK RADAR — MARKET UNIVERSE")
    print("=" * 60)
    print(f"Stocks: {universe.stock_count}")
    print(f"Crypto: {universe.crypto_count}")
    print(f"Total:  {universe.total_count}")
    print("=" * 60)