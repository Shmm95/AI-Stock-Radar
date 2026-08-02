"""Persistent paper-trading account storage for AI-Stock-Radar."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.risk.risk_manager import PortfolioSnapshot


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PAPER_DATA_DIRECTORY = PROJECT_ROOT / "data" / "paper"

ACCOUNT_PATH = PAPER_DATA_DIRECTORY / "account.json"
ORDERS_PATH = PAPER_DATA_DIRECTORY / "orders.json"
POSITIONS_PATH = PAPER_DATA_DIRECTORY / "positions.json"
TRADES_PATH = PAPER_DATA_DIRECTORY / "trades.json"
EQUITY_CURVE_PATH = PAPER_DATA_DIRECTORY / "equity_curve.json"

SCHEMA_VERSION = 1
DEFAULT_INITIAL_CASH = 10_000.0
DEFAULT_CURRENCY = "EUR"


def _utc_now() -> str:
    """Return the current UTC timestamp in ISO-8601 format."""

    return datetime.now(UTC).isoformat()


def _round_money(value: float) -> float:
    """Round a monetary value to two decimal places."""

    return round(float(value), 2)


def _atomic_write_json(
    path: Path,
    payload: Any,
) -> None:
    """Write JSON safely using a temporary file and atomic replacement."""

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        f"{path.suffix}.tmp"
    )

    try:
        with temporary_path.open(
            mode="w",
            encoding="utf-8",
        ) as file:
            json.dump(
                payload,
                file,
                ensure_ascii=False,
                indent=2,
            )

            file.write("\n")
            file.flush()
            os.fsync(file.fileno())

        temporary_path.replace(path)

    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _read_json(
    path: Path,
    default: Any,
) -> Any:
    """Read JSON, returning the supplied default for empty files."""

    if not path.exists():
        return default

    try:
        raw_content = path.read_text(
            encoding="utf-8",
        ).strip()

    except OSError as error:
        raise RuntimeError(
            f"Could not read paper-trading file: {path}"
        ) from error

    if not raw_content:
        return default

    try:
        return json.loads(raw_content)

    except json.JSONDecodeError as error:
        raise ValueError(
            f"Invalid JSON in paper-trading file: {path}"
        ) from error


def _validate_non_negative(
    value: float,
    field_name: str,
) -> None:
    """Validate that a numeric value is not negative."""

    if value < 0:
        raise ValueError(
            f"{field_name} cannot be negative."
        )


@dataclass(frozen=True, slots=True)
class PaperAccount:
    """Persistent state of the local paper-trading account."""

    account_id: str

    currency: str

    initial_cash: float
    cash: float

    realized_pnl: float
    fees_paid: float

    deposits: float
    withdrawals: float

    created_at: str
    updated_at: str

    schema_version: int = SCHEMA_VERSION

    @property
    def net_cash_flow(self) -> float:
        """Return deposits minus withdrawals."""

        return _round_money(
            self.deposits - self.withdrawals
        )

    @property
    def cash_return(self) -> float:
        """Return cash-level performance relative to initial capital."""

        invested_capital = (
            self.initial_cash
            + self.deposits
            - self.withdrawals
        )

        if invested_capital <= 0:
            return 0.0

        return round(
            (
                self.cash
                - invested_capital
            )
            / invested_capital
            * 100,
            4,
        )

    def account_value(
        self,
        positions_market_value: float = 0.0,
    ) -> float:
        """Return cash plus current open-position market value."""

        _validate_non_negative(
            positions_market_value,
            "positions_market_value",
        )

        return _round_money(
            self.cash
            + positions_market_value
        )


class PaperAccountStore:
    """Create, load, update, and reset the local paper account."""

    def __init__(
        self,
        account_path: Path = ACCOUNT_PATH,
    ) -> None:
        self.account_path = account_path

    def initialize_storage(self) -> None:
        """Ensure every paper-trading JSON file contains valid JSON."""

        PAPER_DATA_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        file_defaults = {
            ORDERS_PATH: [],
            POSITIONS_PATH: [],
            TRADES_PATH: [],
            EQUITY_CURVE_PATH: [],
        }

        for path, default in file_defaults.items():
            existing = _read_json(
                path,
                default,
            )

            if existing == default and (
                not path.exists()
                or not path.read_text(
                    encoding="utf-8",
                ).strip()
            ):
                _atomic_write_json(
                    path,
                    default,
                )

    def create(
        self,
        initial_cash: float = DEFAULT_INITIAL_CASH,
        currency: str = DEFAULT_CURRENCY,
        overwrite: bool = False,
    ) -> PaperAccount:
        """Create a new paper account."""

        if initial_cash <= 0:
            raise ValueError(
                "initial_cash must be greater than zero."
            )

        normalized_currency = currency.strip().upper()

        if not normalized_currency:
            raise ValueError(
                "currency cannot be empty."
            )

        existing_content = ""

        if self.account_path.exists():
            existing_content = self.account_path.read_text(
                encoding="utf-8",
            ).strip()

        if existing_content and not overwrite:
            raise FileExistsError(
                "A paper account already exists. "
                "Use load() or create(..., overwrite=True)."
            )

        timestamp = _utc_now()

        account = PaperAccount(
            account_id="paper-primary",
            currency=normalized_currency,
            initial_cash=_round_money(initial_cash),
            cash=_round_money(initial_cash),
            realized_pnl=0.0,
            fees_paid=0.0,
            deposits=0.0,
            withdrawals=0.0,
            created_at=timestamp,
            updated_at=timestamp,
        )

        self.initialize_storage()
        self.save(account)

        return account

    def load(
        self,
        create_if_missing: bool = True,
        initial_cash: float = DEFAULT_INITIAL_CASH,
        currency: str = DEFAULT_CURRENCY,
    ) -> PaperAccount:
        """Load the paper account, optionally creating it when absent."""

        payload = _read_json(
            self.account_path,
            default={},
        )

        if not payload:
            if not create_if_missing:
                raise FileNotFoundError(
                    "Paper account has not been initialized."
                )

            return self.create(
                initial_cash=initial_cash,
                currency=currency,
                overwrite=True,
            )

        required_fields = {
            "account_id",
            "currency",
            "initial_cash",
            "cash",
            "realized_pnl",
            "fees_paid",
            "deposits",
            "withdrawals",
            "created_at",
            "updated_at",
        }

        missing_fields = required_fields.difference(
            payload
        )

        if missing_fields:
            missing_text = ", ".join(
                sorted(missing_fields)
            )

            raise ValueError(
                "account.json is missing required fields: "
                f"{missing_text}"
            )

        account = PaperAccount(
            account_id=str(
                payload["account_id"]
            ),
            currency=str(
                payload["currency"]
            ).upper(),
            initial_cash=_round_money(
                payload["initial_cash"]
            ),
            cash=_round_money(
                payload["cash"]
            ),
            realized_pnl=_round_money(
                payload["realized_pnl"]
            ),
            fees_paid=_round_money(
                payload["fees_paid"]
            ),
            deposits=_round_money(
                payload["deposits"]
            ),
            withdrawals=_round_money(
                payload["withdrawals"]
            ),
            created_at=str(
                payload["created_at"]
            ),
            updated_at=str(
                payload["updated_at"]
            ),
            schema_version=int(
                payload.get(
                    "schema_version",
                    SCHEMA_VERSION,
                )
            ),
        )

        self._validate_account(account)
        self.initialize_storage()

        return account

    def save(
        self,
        account: PaperAccount,
    ) -> None:
        """Validate and persist a paper account."""

        self._validate_account(account)

        updated_account = replace(
            account,
            updated_at=_utc_now(),
        )

        _atomic_write_json(
            self.account_path,
            asdict(updated_account),
        )

    def apply_cash_change(
        self,
        amount: float,
        description: str = "",
    ) -> PaperAccount:
        """Apply a signed cash adjustment and save the account."""

        account = self.load()

        new_cash = _round_money(
            account.cash + amount
        )

        if new_cash < 0:
            raise ValueError(
                "Cash adjustment would make the account negative."
            )

        updated_account = replace(
            account,
            cash=new_cash,
            updated_at=_utc_now(),
        )

        self.save(updated_account)

        if description:
            print(
                f"Cash adjustment: {description} "
                f"({amount:+.2f} {account.currency})"
            )

        return updated_account

    def record_buy(
        self,
        position_value: float,
        fee: float = 0.0,
    ) -> PaperAccount:
        """Deduct the cost and fee of a simulated BUY order."""

        _validate_non_negative(
            position_value,
            "position_value",
        )

        _validate_non_negative(
            fee,
            "fee",
        )

        account = self.load()

        total_cost = _round_money(
            position_value + fee
        )

        if total_cost > account.cash:
            raise ValueError(
                "Paper account does not have enough cash."
            )

        updated_account = replace(
            account,
            cash=_round_money(
                account.cash - total_cost
            ),
            fees_paid=_round_money(
                account.fees_paid + fee
            ),
            updated_at=_utc_now(),
        )

        self.save(updated_account)

        return updated_account

    def record_sell(
        self,
        sale_value: float,
        realized_pnl: float,
        fee: float = 0.0,
    ) -> PaperAccount:
        """Add simulated sale proceeds and realized PnL."""

        _validate_non_negative(
            sale_value,
            "sale_value",
        )

        _validate_non_negative(
            fee,
            "fee",
        )

        account = self.load()

        net_proceeds = _round_money(
            sale_value - fee
        )

        if net_proceeds < 0:
            raise ValueError(
                "Sell fee cannot exceed the sale value."
            )

        updated_account = replace(
            account,
            cash=_round_money(
                account.cash + net_proceeds
            ),
            realized_pnl=_round_money(
                account.realized_pnl
                + realized_pnl
            ),
            fees_paid=_round_money(
                account.fees_paid + fee
            ),
            updated_at=_utc_now(),
        )

        self.save(updated_account)

        return updated_account

    def deposit(
        self,
        amount: float,
    ) -> PaperAccount:
        """Add virtual capital to the paper account."""

        if amount <= 0:
            raise ValueError(
                "Deposit amount must be greater than zero."
            )

        account = self.load()

        updated_account = replace(
            account,
            cash=_round_money(
                account.cash + amount
            ),
            deposits=_round_money(
                account.deposits + amount
            ),
            updated_at=_utc_now(),
        )

        self.save(updated_account)

        return updated_account

    def withdraw(
        self,
        amount: float,
    ) -> PaperAccount:
        """Remove virtual capital from the paper account."""

        if amount <= 0:
            raise ValueError(
                "Withdrawal amount must be greater than zero."
            )

        account = self.load()

        if amount > account.cash:
            raise ValueError(
                "Withdrawal exceeds available paper cash."
            )

        updated_account = replace(
            account,
            cash=_round_money(
                account.cash - amount
            ),
            withdrawals=_round_money(
                account.withdrawals + amount
            ),
            updated_at=_utc_now(),
        )

        self.save(updated_account)

        return updated_account

    def reset(
        self,
        initial_cash: float = DEFAULT_INITIAL_CASH,
        currency: str = DEFAULT_CURRENCY,
    ) -> PaperAccount:
        """Reset the paper environment and erase simulated activity."""

        PAPER_DATA_DIRECTORY.mkdir(
            parents=True,
            exist_ok=True,
        )

        _atomic_write_json(
            ORDERS_PATH,
            [],
        )

        _atomic_write_json(
            POSITIONS_PATH,
            [],
        )

        _atomic_write_json(
            TRADES_PATH,
            [],
        )

        _atomic_write_json(
            EQUITY_CURVE_PATH,
            [],
        )

        return self.create(
            initial_cash=initial_cash,
            currency=currency,
            overwrite=True,
        )

    def create_portfolio_snapshot(
        self,
        *,
        positions_market_value: float = 0.0,
        open_position_count: int = 0,
        open_tickers: tuple[str, ...] = (),
        current_open_risk_amount: float = 0.0,
        current_crypto_value: float = 0.0,
        daily_realized_pnl: float = 0.0,
    ) -> PortfolioSnapshot:
        """Create the input required by the portfolio Risk Manager."""

        account = self.load()

        return PortfolioSnapshot(
            account_value=account.account_value(
                positions_market_value
            ),
            available_cash=account.cash,
            open_position_count=open_position_count,
            open_tickers=open_tickers,
            current_open_risk_amount=current_open_risk_amount,
            current_crypto_value=current_crypto_value,
            daily_realized_pnl=daily_realized_pnl,
        )

    @staticmethod
    def _validate_account(
        account: PaperAccount,
    ) -> None:
        """Validate account state before use or persistence."""

        if account.initial_cash <= 0:
            raise ValueError(
                "Paper account initial_cash must be positive."
            )

        _validate_non_negative(
            account.cash,
            "account.cash",
        )

        _validate_non_negative(
            account.fees_paid,
            "account.fees_paid",
        )

        _validate_non_negative(
            account.deposits,
            "account.deposits",
        )

        _validate_non_negative(
            account.withdrawals,
            "account.withdrawals",
        )

        if not account.currency.strip():
            raise ValueError(
                "Paper account currency cannot be empty."
            )


def print_paper_account(
    account: PaperAccount,
    positions_market_value: float = 0.0,
) -> None:
    """Print a readable paper-account summary."""

    account_value = account.account_value(
        positions_market_value
    )

    print()
    print("=" * 76)
    print("PAPER TRADING ACCOUNT")
    print("=" * 76)

    print(f"Account ID:             {account.account_id}")
    print(f"Currency:               {account.currency}")

    print()
    print(
        f"Initial capital:        "
        f"{account.initial_cash:,.2f} {account.currency}"
    )
    print(
        f"Cash:                   "
        f"{account.cash:,.2f} {account.currency}"
    )
    print(
        f"Open positions value:   "
        f"{positions_market_value:,.2f} {account.currency}"
    )
    print(
        f"Account value:          "
        f"{account_value:,.2f} {account.currency}"
    )

    print()
    print(
        f"Realized PnL:           "
        f"{account.realized_pnl:+,.2f} {account.currency}"
    )
    print(
        f"Fees paid:              "
        f"{account.fees_paid:,.2f} {account.currency}"
    )
    print(
        f"Deposits:               "
        f"{account.deposits:,.2f} {account.currency}"
    )
    print(
        f"Withdrawals:            "
        f"{account.withdrawals:,.2f} {account.currency}"
    )

    print()
    print(f"Created:                {account.created_at}")
    print(f"Updated:                {account.updated_at}")

    print("=" * 76)