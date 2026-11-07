"""Core financial domain models with ownership and money constraints."""

from datetime import date, datetime, timezone
from decimal import Decimal

from flask_login import UserMixin
from werkzeug.security import check_password_hash, generate_password_hash

from app.extensions import db, login_manager


def utc_now() -> datetime:
    """Return a timezone-aware UTC timestamp."""
    return datetime.now(timezone.utc)


class TimestampMixin:
    created_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now)
    updated_at = db.Column(db.DateTime(timezone=True), nullable=False, default=utc_now, onupdate=utc_now)


class User(UserMixin, TimestampMixin, db.Model):
    __tablename__ = "users"

    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(254), nullable=False, unique=True, index=True)
    password_hash = db.Column(db.String(256), nullable=False)
    full_name = db.Column(db.String(120), nullable=False)
    currency = db.Column(db.String(3), nullable=False, default="INR")
    transactions = db.relationship("Transaction", back_populates="user", cascade="all, delete-orphan")
    budgets = db.relationship("Budget", back_populates="user", cascade="all, delete-orphan")
    savings_goals = db.relationship("SavingsGoal", back_populates="user", cascade="all, delete-orphan")
    predictions = db.relationship("Prediction", back_populates="user", cascade="all, delete-orphan")
    recommendations = db.relationship("Recommendation", back_populates="user", cascade="all, delete-orphan")
    chat_history = db.relationship("ChatHistory", back_populates="user", cascade="all, delete-orphan")
    imported_statements = db.relationship("ImportedStatement", back_populates="user", cascade="all, delete-orphan")

    def set_password(self, password: str) -> None:
        if len(password) < 12:
            raise ValueError("Password must contain at least 12 characters.")
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


@login_manager.user_loader
def load_user(user_id: str):
    try:
        return db.session.get(User, int(user_id))
    except (TypeError, ValueError):
        return None


class Category(TimestampMixin, db.Model):
    __tablename__ = "categories"
    __table_args__ = (db.UniqueConstraint("user_id", "name", "transaction_type"),)

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    name = db.Column(db.String(80), nullable=False)
    transaction_type = db.Column(db.String(10), nullable=False)
    is_system = db.Column(db.Boolean, nullable=False, default=False)
    transactions = db.relationship("Transaction", back_populates="category")
    budgets = db.relationship("Budget", back_populates="category")


class Transaction(TimestampMixin, db.Model):
    __tablename__ = "transactions"
    __table_args__ = (
        db.CheckConstraint("amount > 0", name="amount_positive"),
        db.CheckConstraint("transaction_type IN ('income', 'expense')", name="valid_type"),
        db.Index("ix_transactions_user_date", "user_id", "transaction_date"),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    category_id = db.Column(db.Integer, db.ForeignKey("categories.id", ondelete="RESTRICT"), nullable=False, index=True)
    amount = db.Column(db.Numeric(12, 2), nullable=False)
    transaction_type = db.Column(db.String(10), nullable=False)
    description = db.Column(db.String(500), nullable=False, default="")
    notes = db.Column(db.Text, nullable=True)
    transaction_date = db.Column(db.Date, nullable=False, default=date.today, index=True)
    user = db.relationship("User", back_populates="transactions")
    category = db.relationship("Category", back_populates="transactions")


class Budget(TimestampMixin, db.Model):
    __tablename__ = "budgets"
    __table_args__ = (
        db.UniqueConstraint("user_id", "category_id", "period_start", name="uq_budget_user_category_period"),
        db.CheckConstraint("amount > 0", name="amount_positive"),
        db.CheckConstraint("period_end >= period_start", name="valid_period"),
    )

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    category_id = db.Column(db.Integer, db.ForeignKey("categories.id", ondelete="RESTRICT"), nullable=False)
    amount = db.Column(db.Numeric(12, 2), nullable=False)
    name = db.Column(db.String(120), nullable=False, default="Budget", server_default="Budget")
    description = db.Column(db.Text, nullable=True)
    period_start = db.Column(db.Date, nullable=False)
    period_end = db.Column(db.Date, nullable=False)
    user = db.relationship("User", back_populates="budgets")
    category = db.relationship("Category", back_populates="budgets")


class SavingsGoal(TimestampMixin, db.Model):
    __tablename__ = "savings_goals"
    __table_args__ = (db.CheckConstraint("target_amount > 0", name="target_positive"),)

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    name = db.Column(db.String(120), nullable=False)
    target_amount = db.Column(db.Numeric(12, 2), nullable=False)
    target_date = db.Column(db.Date, nullable=True)
    description = db.Column(db.Text, nullable=True)
    completed_at = db.Column(db.DateTime(timezone=True), nullable=True)
    user = db.relationship("User", back_populates="savings_goals")
    contributions = db.relationship("GoalContribution", back_populates="goal", cascade="all, delete-orphan")


class GoalContribution(TimestampMixin, db.Model):
    __tablename__ = "goal_contributions"
    __table_args__ = (db.CheckConstraint("amount > 0", name="amount_positive"),)

    id = db.Column(db.Integer, primary_key=True)
    goal_id = db.Column(db.Integer, db.ForeignKey("savings_goals.id", ondelete="CASCADE"), nullable=False, index=True)
    amount = db.Column(db.Numeric(12, 2), nullable=False)
    contributed_on = db.Column(db.Date, nullable=False, default=date.today)
    note = db.Column(db.String(500), nullable=True)
    goal = db.relationship("SavingsGoal", back_populates="contributions")


class Prediction(TimestampMixin, db.Model):
    __tablename__ = "predictions"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    period_start = db.Column(db.Date, nullable=False)
    predicted_expense = db.Column(db.Numeric(12, 2), nullable=False)
    model_name = db.Column(db.String(80), nullable=False)
    metrics = db.Column(db.JSON, nullable=False, default=dict)
    user = db.relationship("User", back_populates="predictions")


class Recommendation(TimestampMixin, db.Model):
    __tablename__ = "recommendations"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    title = db.Column(db.String(160), nullable=False)
    message = db.Column(db.String(1000), nullable=False)
    rationale = db.Column(db.JSON, nullable=False, default=dict)
    dismissed_at = db.Column(db.DateTime(timezone=True), nullable=True)
    dedupe_key = db.Column(db.String(120), nullable=True, unique=True, index=True)
    user = db.relationship("User", back_populates="recommendations")


class ChatHistory(TimestampMixin, db.Model):
    __tablename__ = "chat_history"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    role = db.Column(db.String(20), nullable=False)
    content = db.Column(db.Text, nullable=False)
    user = db.relationship("User", back_populates="chat_history")


class ImportedStatement(TimestampMixin, db.Model):
    __tablename__ = "imported_statements"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    original_filename = db.Column(db.String(255), nullable=False)
    status = db.Column(db.String(20), nullable=False, default="pending")
    imported_rows = db.Column(db.Integer, nullable=False, default=0)
    error_summary = db.Column(db.Text, nullable=True)
    user = db.relationship("User", back_populates="imported_statements")

