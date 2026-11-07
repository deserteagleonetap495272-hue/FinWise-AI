"use strict";

document.addEventListener("DOMContentLoaded", () => {
  const typeField = document.getElementById("transaction_type");
  const categoryField = document.getElementById("category_id");
  if (!typeField || !categoryField) return;

  const filterCategories = () => {
    const type = typeField.value;
    for (const option of categoryField.options) {
      if (!option.value) continue;
      const visible = option.dataset.type === type;
      option.hidden = !visible;
      if (!visible && option.selected) categoryField.value = "";
    }
  };
  typeField.addEventListener("change", filterCategories);
  filterCategories();
});
