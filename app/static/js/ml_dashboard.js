"use strict";

document.addEventListener("DOMContentLoaded", () => {
  const element = document.getElementById("ml-chart-data");
  const canvas = document.getElementById("ml-prediction-chart");
  if (!element || !canvas || typeof Chart === "undefined") return;
  const data = JSON.parse(element.textContent);
  const currency = new Intl.NumberFormat(undefined, {
    style: "currency", currency: data.currency, maximumFractionDigits: 2,
  });
  new Chart(canvas, {
    type: "line",
    data: { labels: data.labels, datasets: [
      { label: "Actual expenses", data: data.actual.map((value) => value === null ? null : Number(value)), borderColor: "#32c982", backgroundColor: "#32c98222", tension: 0.2, spanGaps: false },
      { label: "Saved predictions", data: data.predicted.map((value) => value === null ? null : Number(value)), borderColor: "#54a7d8", backgroundColor: "#54a7d822", borderDash: [6, 4], tension: 0.2, spanGaps: false },
    ] },
    options: {
      responsive: true, maintainAspectRatio: false,
      interaction: { mode: "index", intersect: false },
      scales: { y: { beginAtZero: true, ticks: { callback: (value) => currency.format(value) } } },
      plugins: { tooltip: { callbacks: { label: (context) => `${context.dataset.label}: ${currency.format(context.parsed.y)}` } } },
    },
  });
});
