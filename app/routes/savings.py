"""Authenticated savings goal and contribution routes."""

from datetime import date

from flask import Blueprint, abort, flash, redirect, render_template, request, url_for
from flask_login import current_user, login_required
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError

from app.extensions import db
from app.forms import ContributionForm, SavingsGoalForm
from app.models import GoalContribution, SavingsGoal
from app.services.savings import (
    add_contribution,
    create_goal,
    delete_contribution,
    goal_overview,
    goal_progress,
    savings_summary,
    update_contribution,
    update_goal,
)

bp = Blueprint("savings", __name__, url_prefix="/savings")


def _owned_goal_or_404(goal_id: int) -> SavingsGoal:
    goal = db.session.scalar(select(SavingsGoal).where(
        SavingsGoal.id == goal_id,
        SavingsGoal.user_id == current_user.id,
    ))
    if goal is None:
        abort(404)
    return goal


def _owned_contribution_or_404(goal_id: int, contribution_id: int) -> GoalContribution:
    contribution = db.session.scalar(select(GoalContribution).where(
        GoalContribution.id == contribution_id,
        GoalContribution.goal_id == goal_id,
    ))
    if contribution is None:
        abort(404)
    return contribution


@bp.get("")
@login_required
def list_goals():
    return render_template("savings/list.html", entries=goal_overview(current_user.id))


@bp.route("/new", methods=["GET", "POST"])
@login_required
def new_goal():
    form = SavingsGoalForm()
    if form.validate_on_submit():
        try:
            create_goal(
                current_user.id, name=form.name.data, target_amount=form.target_amount.data,
                target_date=form.target_date.data, description=form.description.data,
            )
        except ValueError as error:
            flash(str(error), "error")
        except SQLAlchemyError:
            db.session.rollback()
            flash("The savings goal could not be saved. Please try again.", "error")
        else:
            flash("Savings goal created.", "success")
            return redirect(url_for("savings.list_goals"))
    return render_template("savings/goal_form.html", form=form, heading="Create savings goal")


@bp.route("/<int:goal_id>/edit", methods=["GET", "POST"])
@login_required
def edit_goal(goal_id: int):
    goal = _owned_goal_or_404(goal_id)
    form = SavingsGoalForm(obj=goal)
    if form.validate_on_submit():
        try:
            update_goal(
                goal, name=form.name.data, target_amount=form.target_amount.data,
                target_date=form.target_date.data, description=form.description.data,
            )
        except ValueError as error:
            flash(str(error), "error")
        except SQLAlchemyError:
            db.session.rollback()
            flash("The savings goal could not be updated. Please try again.", "error")
        else:
            flash("Savings goal updated.", "success")
            return redirect(url_for("savings.list_goals"))
    return render_template("savings/goal_form.html", form=form, heading="Edit savings goal")


@bp.post("/<int:goal_id>/delete")
@login_required
def delete_goal(goal_id: int):
    goal = _owned_goal_or_404(goal_id)
    try:
        db.session.delete(goal)
        db.session.commit()
    except SQLAlchemyError:
        db.session.rollback()
        flash("The savings goal could not be deleted. Please try again.", "error")
    else:
        flash("Savings goal deleted.", "success")
    return redirect(url_for("savings.list_goals"))


@bp.route("/<int:goal_id>/contributions", methods=["GET", "POST"])
@login_required
def contributions(goal_id: int):
    goal = _owned_goal_or_404(goal_id)
    form = ContributionForm()
    if request.method == "GET" and not form.contributed_on.data:
        form.contributed_on.data = date.today()
    if form.validate_on_submit():
        try:
            add_contribution(goal, amount=form.amount.data, contributed_on=form.contributed_on.data,
                             note=form.note.data)
        except ValueError as error:
            flash(str(error), "error")
        except SQLAlchemyError:
            db.session.rollback()
            flash("The contribution could not be saved. Please try again.", "error")
        else:
            flash("Contribution added.", "success")
            return redirect(url_for("savings.contributions", goal_id=goal.id))
    return render_template(
        "savings/contributions.html", goal=goal, form=form,
        progress=goal_progress(goal),
        contributions=db.session.scalars(select(GoalContribution).where(
            GoalContribution.goal_id == goal.id,
        ).order_by(GoalContribution.contributed_on.desc(), GoalContribution.id.desc())).all(),
    )


@bp.route("/<int:goal_id>/contributions/<int:contribution_id>/edit", methods=["GET", "POST"])
@login_required
def edit_contribution(goal_id: int, contribution_id: int):
    goal = _owned_goal_or_404(goal_id)
    contribution = _owned_contribution_or_404(goal.id, contribution_id)
    form = ContributionForm(obj=contribution)
    if form.validate_on_submit():
        try:
            update_contribution(
                goal, contribution, amount=form.amount.data,
                contributed_on=form.contributed_on.data, note=form.note.data,
            )
        except ValueError as error:
            flash(str(error), "error")
        except SQLAlchemyError:
            db.session.rollback()
            flash("The contribution could not be updated. Please try again.", "error")
        else:
            flash("Contribution updated.", "success")
            return redirect(url_for("savings.contributions", goal_id=goal.id))
    return render_template("savings/contribution_form.html", goal=goal, form=form, heading="Edit contribution")


@bp.post("/<int:goal_id>/contributions/<int:contribution_id>/delete")
@login_required
def remove_contribution(goal_id: int, contribution_id: int):
    goal = _owned_goal_or_404(goal_id)
    contribution = _owned_contribution_or_404(goal.id, contribution_id)
    try:
        delete_contribution(goal, contribution)
    except ValueError as error:
        flash(str(error), "error")
    except SQLAlchemyError:
        db.session.rollback()
        flash("The contribution could not be deleted. Please try again.", "error")
    else:
        flash("Contribution deleted.", "success")
    return redirect(url_for("savings.contributions", goal_id=goal.id))
