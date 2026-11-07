"""User-scoped transaction operations and financial summaries."""

from calendar import monthrange
from datetime import date
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError

from app.extensions import db
from app.models import Category, Transaction

ZERO = Decimal("0.00")
CATEGORY_DEFAULTS = {
    "income": ("Salary", "Freelancing", "Business", "Investments", "Gifts", "Other Income"),
    "expense": (
        "Food", "Groceries", "Transportation", "Shopping", "Education", "Healthcare",
        "Entertainment", "Rent", "Bills", "Subscriptions", "Travel", "Other Expense",
    ),
}
CURRENCY_SYMBOLS = {"INR": "₹", "USD": "$", "EUR": "€", "GBP": "£", "JPY": "¥", "CAD": "CA$", "AUD": "A$"}


def format_money(value: Decimal, currency: str = "INR") -> str:
    """Format a Decimal amount with the user's currency code or symbol."""
    symbol = CURRENCY_SYMBOLS.get(currency.upper())
    prefix = symbol if symbol else currency.upper()
    return f"{prefix} {Decimal(value):,.2f}"


class InvalidCategoryError(ValueError):
    """Raised when a category is not owned by the user or has the wrong type."""


def ensure_default_categories(user_id: int) -> None:
    """Create missing defaults owned by this user, without duplicates."""
    existing = set(db.session.execute(
        select(Category.name, Category.transaction_type).where(Category.user_id == user_id)
    ).all())
    missing = [
        Category(user_id=user_id, name=name, transaction_type=kind, is_system=True)
        for kind, names in CATEGORY_DEFAULTS.items()
        for name in names
        if (name, kind) not in existing
    ]
    if not missing:
        return
    db.session.add_all(missing)
    try:
        db.session.commit()
    except IntegrityError:
        # A concurrent request may have initialized the same user's defaults.
        db.session.rollback()
        existing = set(db.session.execute(
            select(Category.name, Category.transaction_type).where(Category.user_id == user_id)
        ).all())
        if any((name, kind) not in existing for kind, names in CATEGORY_DEFAULTS.items() for name in names):
            raise


def get_user_categories(user_id: int) -> list[Category]:
    """Read only this user's categories; defaults are initialized on POST flows."""
    return list(db.session.scalars(
        select(Category).where(Category.user_id == user_id).order_by(Category.transaction_type, Category.name)
    ))


def _validated_category(user_id: int, category_id: int, transaction_type: str) -> Category:
    category = db.session.scalar(select(Category).where(
        Category.id == category_id,
        Category.user_id == user_id,
        Category.transaction_type == transaction_type,
    ))
    if category is None:
        raise InvalidCategoryError("Choose a valid category for this transaction type.")
    return category


def create_transaction(user_id: int, *, amount: Decimal, transaction_type: str,
                       category_id: int, description: str, transaction_date: date,
                       notes: str | None = None) -> Transaction:
    """Create a transaction after validating category ownership and type."""
    category = _validated_category(user_id, category_id, transaction_type)
    transaction = Transaction(
        user_id=user_id, category=category, amount=amount, transaction_type=transaction_type,
        description=description, transaction_date=transaction_date, notes=notes or None,
    )
    db.session.add(transaction)
    db.session.commit()
    return transaction


def update_transaction(transaction: Transaction, *, user_id: int, amount: Decimal,
                       transaction_type: str, category_id: int, description: str,
                       transaction_date: date, notes: str | None = None) -> Transaction:
    """Update transaction fields while preserving its owner."""
    category = _validated_category(user_id, category_id, transaction_type)
    transaction.category = category
    transaction.amount = amount
    transaction.transaction_type = transaction_type
    transaction.description = description
    transaction.transaction_date = transaction_date
    transaction.notes = notes or None
    db.session.commit()
    return transaction


def financial_summary(user_id: int, today: date | None = None) -> dict:
    """Return exact decimal totals and chart series from one user's records."""
    today = today or date.today()
    month_start = today.replace(day=1)
    month_end = today.replace(day=monthrange(today.year, today.month)[1])

    def total(transaction_type: str, start: date | None = None, end: date | None = None) -> Decimal:
        query = select(func.sum(Transaction.amount)).where(
            Transaction.user_id == user_id,
            Transaction.transaction_type == transaction_type,
        )
        if start:
            query = query.where(Transaction.transaction_date >= start)
        if end:
            query = query.where(Transaction.transaction_date <= end)
        return db.session.scalar(query) or ZERO

    total_income = total("income")
    total_expenses = total("expense")
    monthly_income = total("income", month_start, month_end)
    monthly_expenses = total("expense", month_start, month_end)

    month_starts = []
    year, month = today.year, today.month
    for _ in range(6):
        month_starts.append(date(year, month, 1))
        month -= 1
        if month == 0:
            year, month = year - 1, 12
    month_starts.reverse()
    trend_start = month_starts[0]
    trend_transactions = db.session.scalars(select(Transaction).where(
        Transaction.user_id == user_id,
        Transaction.transaction_date >= trend_start,
        Transaction.transaction_date <= today,
    ))
    month_values = {start: {"income": ZERO, "expense": ZERO} for start in month_starts}
    for transaction in trend_transactions:
        key = transaction.transaction_date.replace(day=1)
        if key in month_values:
            month_values[key][transaction.transaction_type] += transaction.amount

    category_rows = db.session.execute(
        select(Category.name, func.sum(Transaction.amount))
        .join(Transaction, Transaction.category_id == Category.id)
        .where(
            Transaction.user_id == user_id,
            Transaction.transaction_type == "expense",
            Transaction.transaction_date >= month_start,
            Transaction.transaction_date <= month_end,
            Category.user_id == user_id,
        )
        .group_by(Category.name)
        .order_by(func.sum(Transaction.amount).desc())
    ).all()

    return {
        "total_income": total_income,
        "total_expenses": total_expenses,
        "balance": total_income - total_expenses,
        "total_savings": total_income - total_expenses,
        "monthly_income": monthly_income,
        "monthly_expenses": monthly_expenses,
        "monthly_savings": monthly_income - monthly_expenses,
        "monthly_net_cash_flow": monthly_income - monthly_expenses,
        "trend_labels": [start.strftime("%b %Y") for start in month_starts],
        "trend_income": [str(month_values[start]["income"]) for start in month_starts],
        "trend_expenses": [str(month_values[start]["expense"]) for start in month_starts],
        "category_labels": [row[0] for row in category_rows],
        "category_values": [str(row[1] or ZERO) for row in category_rows],
    }


def recent_transactions(user_id: int, limit: int = 5) -> list[Transaction]:
    """Return a user's latest transactions only."""
    return list(db.session.scalars(
        select(Transaction)
        .where(Transaction.user_id == user_id)
        .order_by(Transaction.transaction_date.desc(), Transaction.created_at.desc(), Transaction.id.desc())
        .limit(limit)
    ))
