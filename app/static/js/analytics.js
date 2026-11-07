"use strict";

document.addEventListener("DOMContentLoaded", () => {
  const element = document.getElementById("analytics-chart-data");
  if (!element || typeof Chart === "undefined") return;
  const data = JSON.parse(element.textContent);
  const currency = new Intl.NumberFormat(undefined, { style: "currency", currency: data.currency, maximumFractionDigits: 2 });
  const grouped = data.selected_period === "yearly"
    ? { labels: data.yearly_labels, income: data.yearly_income, expenses: data.yearly_expenses }
    : { labels: data.monthly_labels, income: data.monthly_income, expenses: data.monthly_expenses };
  const moneyTooltip = (axis) => ({ callbacks: { label: (context) => `${context.dataset.label}: ${currency.format(axis === "x" ? context.parsed.x : context.parsed.y)}` } });

  const incomeExpense = document.getElementById("analytics-income-expense");
  if (incomeExpense) new Chart(incomeExpense, {
    type: "bar", data: { labels: grouped.labels, datasets: [
      { label: "Income", data: grouped.income.map(Number), backgroundColor: "#32c982" },
      { label: "Expenses", data: grouped.expenses.map(Number), backgroundColor: "#e47c78" },
    ] }, options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true } }, plugins: { tooltip: moneyTooltip("y") } },
  });

  const trend = document.getElementById("analytics-spending-trend");
  if (trend) new Chart(trend, {
    type: "line", data: { labels: grouped.labels, datasets: [{ label: "Expenses", data: grouped.expenses.map(Number), borderColor: "#32c982", tension: 0.25, fill: true, backgroundColor: "#32c98222" }] },
    options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true } }, plugins: { tooltip: moneyTooltip("y") } },
  });

  const category = document.getElementById("analytics-category");
  if (category) new Chart(category, {
    type: "doughnut", data: { labels: data.category_labels, datasets: [{ data: data.category_expenses.map(Number), backgroundColor: ["#32c982", "#54a7d8", "#a18be0", "#e6b85c", "#e47c78", "#5ac0ae", "#7998ba", "#ca82aa"] }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { tooltip: { callbacks: { label: (context) => `${context.label}: ${currency.format(context.parsed)}` } } } },
  });

  const weekly = document.getElementById("analytics-weekly");
  if (weekly) new Chart(weekly, {
    type: "bar", data: { labels: data.weekly_labels, datasets: [{ label: "Expenses", data: data.weekly_expenses.map(Number), backgroundColor: "#54a7d8" }] },
    options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true } }, plugins: { tooltip: moneyTooltip("y") } },
  });

  const savings = document.getElementById("analytics-savings-rate");
  if (savings) new Chart(savings, {
    type: "line", data: { labels: data.savings_rate_labels, datasets: [{ label: "Savings rate", data: data.savings_rates.map((value) => value === null ? null : Number(value)), borderColor: "#a18be0", tension: 0.25 }] },
    options: { responsive: true, maintainAspectRatio: false, scales: { y: { ticks: { callback: (value) => `${value}%` } } }, plugins: { tooltip: { callbacks: { label: (context) => `${context.parsed.y === null ? "No income" : `${context.parsed.y.toFixed(1)}%`}` } } } },
  });
});
