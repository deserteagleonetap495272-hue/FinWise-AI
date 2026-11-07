"""Validated web forms for account registration and sign-in."""

import re
from decimal import Decimal

from flask_wtf import FlaskForm
from wtforms import (
    DateField,
    DecimalField,
    PasswordField,
    SelectField,
    SelectChoice,
    StringField,
    SubmitField,
    TextAreaField,
)
from wtforms.validators import DataRequired, EqualTo, Length, NumberRange, ValidationError


def validate_email_address(_form, field) -> None:
    """Apply a small syntax check without requiring an external email service."""
    value = field.data.strip().lower()
    if len(value) > 254 or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", value):
        raise ValidationError("Enter a valid email address.")


class OptionalDateField(DateField):
    """A date field that maps an empty browser date input to ``None``."""

    def process_formdata(self, valuelist) -> None:
        if valuelist and not valuelist[0].strip():
            self.data = None
            return
        super().process_formdata(valuelist)


class RegistrationForm(FlaskForm):
    """Collect and validate the fields needed to create an account."""

    full_name = StringField(
        "Full name", filters=[lambda value: value.strip() if value else value],
        validators=[DataRequired(), Length(min=2, max=120)],
    )
    email = StringField(
        "Email", filters=[lambda value: value.strip().lower() if value else value],
        validators=[DataRequired(), validate_email_address],
    )
    password = PasswordField(
        "Password", validators=[
            DataRequired(),
            Length(min=12, max=128, message="Password must be between 12 and 128 characters."),
        ],
    )
    confirm_password = PasswordField(
        "Confirm password",
        validators=[DataRequired(), EqualTo("password", message="Passwords must match.")],
    )
    submit = SubmitField("Create account")


class LoginForm(FlaskForm):
    """Collect credentials for password authentication."""

    email = StringField(
        "Email", filters=[lambda value: value.strip().lower() if value else value],
        validators=[DataRequired(), validate_email_address],
    )
    password = PasswordField("Password", validators=[DataRequired(), Length(max=128)])
    submit = SubmitField("Log in")


class TransactionForm(FlaskForm):
    """Input validation for creating and editing a financial transaction."""

    transaction_type = SelectField(
        "Type", choices=[SelectChoice("income", "Income"), SelectChoice("expense", "Expense")],
        validators=[DataRequired()],
    )
    amount = DecimalField(
        "Amount", places=2, rounding=None,
        validators=[DataRequired(), NumberRange(
            min=Decimal("0.01"), max=Decimal("9999999999.99"),
            message="Amount must be between 0.01 and 9,999,999,999.99.",
        )],
    )
    category_id = SelectField("Category", coerce=int, validators=[DataRequired()])
    description = StringField(
        "Description", filters=[lambda value: value.strip() if value else value],
        validators=[DataRequired(), Length(max=500)],
    )
    transaction_date = DateField("Date", format="%Y-%m-%d", validators=[DataRequired()])
    notes = TextAreaField(
        "Notes (optional)", filters=[lambda value: value.strip() or None if value else None],
        validators=[Length(max=5000)],
    )
    submit = SubmitField("Save transaction")


class BudgetForm(FlaskForm):
    """Validate a category budget and its period."""

    name = StringField(
        "Budget name", filters=[lambda value: value.strip() if value else value],
        validators=[DataRequired(), Length(max=120)],
    )
    category_id = SelectField("Expense category", coerce=int, validators=[DataRequired()])
    amount = DecimalField(
        "Budget amount", places=2,
        validators=[DataRequired(), NumberRange(
            min=Decimal("0.01"), max=Decimal("9999999999.99"),
            message="Amount must be between 0.01 and 9,999,999,999.99.",
        )],
    )
    period_type = SelectField(
        "Budget period", choices=[SelectChoice("monthly", "Monthly"), SelectChoice("custom", "Custom date range")],
        validators=[DataRequired()],
    )
    period_start = DateField("Start date", format="%Y-%m-%d", validators=[DataRequired()])
    period_end = DateField("End date", format="%Y-%m-%d", validators=[DataRequired()])
    description = TextAreaField(
        "Description (optional)", filters=[lambda value: value.strip() or None if value else None],
        validators=[Length(max=2000)],
    )
    submit = SubmitField("Save budget")


class SavingsGoalForm(FlaskForm):
    """Validate goal target and optional deadline."""

    name = StringField(
        "Goal name", filters=[lambda value: value.strip() if value else value],
        validators=[DataRequired(), Length(max=120)],
    )
    target_amount = DecimalField(
        "Target amount", places=2,
        validators=[DataRequired(), NumberRange(
            min=Decimal("0.01"), max=Decimal("9999999999.99"),
            message="Amount must be between 0.01 and 9,999,999,999.99.",
        )],
    )
    target_date = OptionalDateField("Target date (optional)", format="%Y-%m-%d", validators=[])
    description = TextAreaField(
        "Description (optional)", filters=[lambda value: value.strip() or None if value else None],
        validators=[Length(max=2000)],
    )
    submit = SubmitField("Save goal")


class ContributionForm(FlaskForm):
    """Validate an individual goal contribution."""

    amount = DecimalField(
        "Contribution amount", places=2,
        validators=[DataRequired(), NumberRange(
            min=Decimal("0.01"), max=Decimal("9999999999.99"),
            message="Amount must be between 0.01 and 9,999,999,999.99.",
        )],
    )
    contributed_on = DateField("Contribution date", format="%Y-%m-%d", validators=[DataRequired()])
    note = StringField(
        "Note (optional)", filters=[lambda value: value.strip() or None if value else None],
        validators=[Length(max=500)],
    )
    submit = SubmitField("Save contribution")
