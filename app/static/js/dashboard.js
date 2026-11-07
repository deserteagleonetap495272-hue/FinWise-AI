"use strict";

document.addEventListener("DOMContentLoaded", () => {
  const dataElement = document.getElementById("dashboard-chart-data");
  if (!dataElement || typeof Chart === "undefined") return;
  const data = JSON.parse(dataElement.textContent);
  const formatCurrency = new Intl.NumberFormat(undefined, {
    style: "currency", currency: data.currency, maximumFractionDigits: 2,
  });
  const moneyOptions = { callbacks: { label: (context) => formatCurrency.format(context.parsed.y ?? context.parsed) } };

  const incomeExpenseCanvas = document.getElementById("income-expenses-chart");
  if (incomeExpenseCanvas) new Chart(incomeExpenseCanvas, {
    type: "bar",
    data: { labels: ["This month"], datasets: [
      { label: "Income", data: [Number(data.monthly_income)], backgroundColor: "#32c982" },
      { label: "Expenses", data: [Number(data.monthly_expenses)], backgroundColor: "#e47c78" },
    ] },
    options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true } }, plugins: { tooltip: moneyOptions } },
  });

  const trendCanvas = document.getElementById("spending-trend-chart");
  if (trendCanvas) new Chart(trendCanvas, {
    type: "line",
    data: { labels: data.labels, datasets: [
      { label: "Income", data: data.trend_income.map(Number), borderColor: "#32c982", tension: 0.3 },
      { label: "Expenses", data: data.trend_expenses.map(Number), borderColor: "#e47c78", tension: 0.3 },
    ] },
    options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true } }, plugins: { tooltip: moneyOptions } },
  });

  const categoryCanvas = document.getElementById("category-chart");
  if (categoryCanvas && data.category_labels.length) new Chart(categoryCanvas, {
    type: "doughnut",
    data: { labels: data.category_labels, datasets: [{ data: data.category_values.map(Number), backgroundColor: ["#32c982", "#54a7d8", "#a18be0", "#e6b85c", "#e47c78", "#5ac0ae", "#7998ba", "#ca82aa"] }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { tooltip: { callbacks: { label: (context) => `${context.label}: ${formatCurrency.format(context.parsed)}` } } } },
  });

  const budgetCanvas = document.getElementById("budget-chart");
  if (budgetCanvas) new Chart(budgetCanvas, {
    type: "bar",
    data: { labels: data.budget_names, datasets: [
      { label: "Budget", data: data.budget_amounts.map(Number), backgroundColor: "#54a7d8" },
      { label: "Actual", data: data.budget_spent.map(Number), backgroundColor: "#32c982" },
    ] },
    options: { indexAxis: "y", responsive: true, maintainAspectRatio: false, scales: { x: { beginAtZero: true } }, plugins: { tooltip: { callbacks: { label: (context) => `${context.dataset.label}: ${formatCurrency.format(context.parsed.x)}` } } } },
  });

  const utilizationCanvas = document.getElementById("budget-utilization-chart");
  if (utilizationCanvas) new Chart(utilizationCanvas, {
    type: "doughnut",
    data: { labels: data.budget_names, datasets: [{ data: data.budget_utilization.map(Number), backgroundColor: ["#32c982", "#54a7d8", "#a18be0", "#e6b85c", "#e47c78", "#5ac0ae"] }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { tooltip: { callbacks: { label: (context) => `${context.label}: ${context.parsed.toFixed(1)}%` } } } },
  });

  const savingsCanvas = document.getElementById("savings-chart");
  if (savingsCanvas) new Chart(savingsCanvas, {
    type: "bar",
    data: { labels: data.savings_names, datasets: [
      { label: "Target", data: data.savings_targets.map(Number), backgroundColor: "#54a7d8" },
      { label: "Saved", data: data.savings_saved.map(Number), backgroundColor: "#32c982" },
    ] },
    options: { indexAxis: "y", responsive: true, maintainAspectRatio: false, scales: { x: { beginAtZero: true } }, plugins: { tooltip: { callbacks: { label: (context) => `${context.dataset.label}: ${formatCurrency.format(context.parsed.x)}` } } } },
  });
});
