"""Reproducible, fictional INR transactions for Stage 9 validation."""

from __future__ import annotations

from calendar import monthrange
from dataclasses import dataclass
from datetime import date
from decimal import Decimal
import random

SEED = 20260917
MONTHS = 24
START_YEAR = 2024
START_MONTH = 1
CENT = Decimal("0.01")

USERS = (
    {"key": "primary", "email": "synthetic.primary@example.test", "full_name": "Synthetic Primary"},
    {"key": "secondary", "email": "synthetic.secondary@example.test", "full_name": "Synthetic Secondary"},
    {"key": "scenario", "email": "synthetic.scenario@example.test", "full_name": "Synthetic Scenario"},
)

EXPENSE_DESCRIPTIONS = {
    "Rent": ("monthly apartment rent transfer", "home rental payment to landlord", "apartment lease rent payment"),
    "Utilities": ("electricity bill payment", "home broadband and power bill", "internet and electricity bill"),
    "Groceries": ("weekly vegetables and pantry shopping", "grocery supermarket household staples", "fresh produce and staples market"),
    "Food delivery": ("food delivery dinner order", "restaurant takeaway meal order", "quick service food app order"),
    "Transportation": ("metro card recharge commute", "bus and auto ride commute", "local train and cab fare"),
    "Education": ("online course tuition payment", "college class fee installment", "certification learning platform fee"),
    "Healthcare": ("pharmacy prescription medicine", "clinic consultation and medicines", "healthcare diagnostic test payment"),
    "Entertainment": ("cinema tickets and streaming", "music and movie subscription", "weekend event entertainment"),
    "Shopping": ("clothing and household shopping", "online retail purchase", "electronics store shopping"),
    "Travel": ("intercity rail travel booking", "hotel and transport trip expense", "family travel reservation"),
    "Bank charges": ("bank account service fee", "payment transaction service charge"),
}
INCOME_DESCRIPTIONS = {
    "Salary": ("monthly salary payroll credit", "employer salary deposit", "monthly payroll transfer"),
    "Freelancing": ("freelance design project payment", "contract writing invoice credit", "independent consulting fee"),
    "Internship stipend": ("monthly internship stipend credit", "training stipend deposit", "intern program payment"),
}


@dataclass(frozen=True)
class SyntheticDataset:
    transactions: tuple[dict, ...]
    contributions: tuple[dict, ...]
    users: tuple[dict, ...] = USERS
    seed: int = SEED

    @property
    def start_date(self) -> date:
        return date(START_YEAR, START_MONTH, 1)

    @property
    def end_date(self) -> date:
        year = START_YEAR + (START_MONTH - 1 + MONTHS - 1) // 12
        month = (START_MONTH - 1 + MONTHS - 1) % 12 + 1
        return date(year, month, monthrange(year, month)[1])


def _money(rng: random.Random, low: int, high: int) -> Decimal:
    return (Decimal(rng.randint(low, high)) / Decimal(100)).quantize(CENT)


def _record(rows: list[dict], user_key: str, category: str, kind: str,
            amount: Decimal, day: date, description: str) -> None:
    rows.append({
        "id": len(rows) + 1,
        "user_key": user_key,
        "transaction_date": day,
        "transaction_type": kind,
        "category": category,
        "category_type": kind,
        "amount": amount.quantize(CENT),
        "description": description,
    })


def build_dataset() -> SyntheticDataset:
    """Build two users' 24-month synthetic ledger and savings contributions."""
    rng = random.Random(SEED)
    transactions: list[dict] = []
    contributions: list[dict] = []

    for offset in range(MONTHS):
        year = START_YEAR + (START_MONTH - 1 + offset) // 12
        month = (START_MONTH - 1 + offset) % 12 + 1
        last_day = monthrange(year, month)[1]
        first = date(year, month, 1)
        last = date(year, month, last_day)
        variant = offset % 3

        # Primary account: ₹50,000 monthly salary and intermittent contract income.
        _record(transactions, "primary", "Salary", "income", Decimal("50000.00"), last,
                INCOME_DESCRIPTIONS["Salary"][variant])
        if offset % 2 == 1:
            _record(transactions, "primary", "Freelancing", "income",
                    _money(rng, 350000, 1450000), date(year, month, min(26, last_day)),
                    INCOME_DESCRIPTIONS["Freelancing"][variant])

        primary_expenses = [
            ("Rent", Decimal("18000.00"), first),
            ("Utilities", _money(rng, 180000, 335000) + (Decimal("650.00") if month in {4, 5, 6, 7, 8} else Decimal("0.00")), date(year, month, 4)),
            ("Groceries", _money(rng, 590000, 940000), date(year, month, 8)),
            ("Food delivery", _money(rng, 95000, 310000), date(year, month, 15)),
            ("Transportation", _money(rng, 140000, 325000), date(year, month, 20)),
            ("Education", Decimal("3500.00"), date(year, month, 5)),
            ("Entertainment", _money(rng, 80000, 320000), last),
        ]
        if offset % 2 == 0:
            primary_expenses.append(("Healthcare", _money(rng, 55000, 230000), date(year, month, 18)))
        if offset % 3 == 0:
            primary_expenses.append(("Shopping", _money(rng, 200000, 850000), date(year, month, 25)))
        if offset % 6 == 2:
            primary_expenses.append(("Travel", _money(rng, 750000, 2200000), date(year, month, 27)))
        if offset in {2, 15}:
            primary_expenses.append(("Bank charges", _money(rng, 1500, 5000), date(year, month, 28)))

        # Unusual but legitimate spending for anomaly-review scenarios.
        if (year, month) == (2025, 7):
            primary_expenses.append(("Shopping", Decimal("72000.00"), date(year, month, 29)))
        if (year, month) == (2024, 10):
            primary_expenses.append(("Healthcare", Decimal("42500.00"), date(year, month, 19)))
        if (year, month) == (2025, 3):
            primary_expenses.append(("Food delivery", Decimal("16500.00"), date(year, month, 16)))

        for category, amount, day in primary_expenses:
            phrases = EXPENSE_DESCRIPTIONS[category]
            description = phrases[(variant + rng.randrange(len(phrases))) % len(phrases)]
            if category == "Shopping" and amount == Decimal("72000.00"):
                description = "one time laptop purchase electronics store"
            elif category == "Healthcare" and amount == Decimal("42500.00"):
                description = "hospital diagnostic and treatment invoice"
            elif category == "Food delivery" and amount == Decimal("16500.00"):
                description = "family celebration catering and dinner order"
            _record(transactions, "primary", category, "expense", amount, day, description)

        # A second, lower-income household exercises per-user data isolation.
        _record(transactions, "secondary", "Internship stipend", "income", Decimal("32000.00"), first,
                INCOME_DESCRIPTIONS["Internship stipend"][variant])
        if offset % 3 == 2:
            _record(transactions, "secondary", "Freelancing", "income",
                    _money(rng, 250000, 900000), date(year, month, min(24, last_day)),
                    INCOME_DESCRIPTIONS["Freelancing"][(variant + 1) % 3])
        secondary_expenses = [
            ("Rent", Decimal("11000.00"), first),
            ("Utilities", _money(rng, 105000, 205000), date(year, month, 4)),
            ("Groceries", _money(rng, 420000, 690000), date(year, month, 9)),
            ("Food delivery", _money(rng, 45000, 175000), date(year, month, 15)),
            ("Transportation", _money(rng, 85000, 210000), date(year, month, 20)),
            ("Education", Decimal("2500.00"), date(year, month, 5)),
            ("Entertainment", _money(rng, 45000, 180000), date(year, month, 23)),
        ]
        if offset % 4 == 1:
            secondary_expenses.append(("Healthcare", _money(rng, 40000, 145000), date(year, month, 18)))
        if offset % 4 == 3:
            secondary_expenses.append(("Shopping", _money(rng, 120000, 450000), date(year, month, 25)))
        for category, amount, day in secondary_expenses:
            phrases = EXPENSE_DESCRIPTIONS[category]
            description = phrases[(variant + rng.randrange(len(phrases))) % len(phrases)]
            _record(transactions, "secondary", category, "expense", amount, day, description)

        contributions.extend((
            {"user_key": "primary", "goal_name": "Emergency savings", "target_amount": Decimal("60000.00"),
             "amount": Decimal("2500.00"), "contributed_on": last, "note": "Monthly savings contribution"},
            {"user_key": "secondary", "goal_name": "Education reserve", "target_amount": Decimal("30000.00"),
             "amount": Decimal("1000.00"), "contributed_on": last, "note": "Monthly savings contribution"},
        ))

        # A separate, deliberately simple 12-month scenario mirrors the
        # documented cash-flow example and is never based on user records.
        if offset < int(SCENARIO["scenario_months"]):
            _record(transactions, "scenario", "Salary", "income", SCENARIO["monthly_income"], last,
                    "synthetic monthly salary scenario")
            _record(transactions, "scenario", "Fixed expenses", "expense", SCENARIO["fixed_expenses"], first,
                    "synthetic monthly fixed household costs")
            _record(transactions, "scenario", "Variable expenses", "expense", SCENARIO["variable_expenses"],
                    date(year, month, min(15, last_day)), "synthetic monthly discretionary costs")
            contributions.append({
                "user_key": "scenario", "goal_name": "12-month savings scenario",
                "target_amount": SCENARIO["savings_goal_target"],
                "amount": SCENARIO["monthly_contribution"], "contributed_on": last,
                "note": "Synthetic monthly savings contribution",
            })

    transactions.sort(key=lambda row: (row["transaction_date"], row["user_key"], row["id"]))
    for row_id, row in enumerate(transactions, start=1):
        row["id"] = row_id
    return SyntheticDataset(tuple(transactions), tuple(contributions))


SCENARIO = {
    "monthly_income": Decimal("50000.00"),
    "fixed_expenses": Decimal("20000.00"),
    "variable_expenses": Decimal("10000.00"),
    "savings_goal_target": Decimal("60000.00"),
    "monthly_contribution": Decimal("5000.00"),
    "scenario_months": 12,
}
