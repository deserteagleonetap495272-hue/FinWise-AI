"use strict";

document.addEventListener("DOMContentLoaded", () => {
  const period = document.getElementById("period_type");
  const start = document.getElementById("period_start");
  const end = document.getElementById("period_end");
  if (!period || !start || !end) return;

  const syncMonthlyDates = () => {
    const isMonthly = period.value === "monthly";
    end.readOnly = isMonthly;
    if (isMonthly && start.value) {
      const [year, month] = start.value.split("-").map(Number);
      const lastDay = new Date(year, month, 0).getDate();
      end.value = `${year}-${String(month).padStart(2, "0")}-${String(lastDay).padStart(2, "0")}`;
    }
  };
  period.addEventListener("change", syncMonthlyDates);
  start.addEventListener("change", syncMonthlyDates);
  syncMonthlyDates();
});
