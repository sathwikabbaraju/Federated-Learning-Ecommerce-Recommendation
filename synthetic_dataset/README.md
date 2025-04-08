# 🧠 Synthetic Data Generation for Buyer Personas

Welcome to the `synthetic_data/` directory – a curated space where we simulate **realistic buyer behavior** through meticulously defined **personas** to support Federated Learning for personalized e-commerce recommendations.

---

## 📌 Project Overview

This synthetic dataset was designed and implemented by **Sathwik**, the Team Lead & Model Architect for the Federated Learning E-Commerce Recommendation (FLEC) system. The purpose of this dataset is to model different types of **consumer decision-making behaviors** under diverse shopping conditions using **mathematical logic**, **statistical heuristics**, and **domain expertise**.

Each persona simulates purchases from a cleaned version of the [Amazon Product Sales Dataset 2023](https://example.com) with the following columns:

```
name, main_category, sub_category, image, link, rating, no_of_ratings, discount_price, actual_price
```

---

## 👤 The 11 Buyer Personas

Each persona file (`persona_XX_<name>.py`) generates synthetic purchase decisions based on a unique shopping philosophy:

| Persona ID | Name           | Logic |
|------------|----------------|-------|
| 1 | `ratermax`     | High rating (≥ 4.0) + Highest actual price |
| 2 | `ratermed`     | Medium rating (3.0–3.9) + Highest actual price |
| 3 | `impulsivemax` | Ignores rating + Highest actual price |
| 4 | `valuerater`   | High rating + Medium discount (discount < actual) |
| 5 | `midvalue`     | Medium rating + Medium discount |
| 6 | `impulsivemid` | Ignores rating + Medium discount |
| 7 | `middiscount`  | Medium rating + Sort by discount price |
| 8 | `saverpro`     | High rating + Max discount % |
| 9 | `savermed`     | Medium rating + Max discount % |
|10 | `saverfree`    | Ignores rating + Max discount % |
|11 | `socialproof`  | Highest (rating × no_of_ratings) |

Each script filters the product catalog **by category**, applies **mathematical filters and sorting**, and selects representative products to simulate a "purchase".

---

## 🧮 Mathematical Foundation

Some of the formulas used:

- **Score for popularity-based decisions**:  
  `score = rating × no_of_ratings`

- **Discount percentage logic**:  
  `discount_percent = (actual_price - discount_price) / actual_price`

- **Filtering logic for categories and rating bands**:  
  Custom pandas filters applied during group-wise operations.

---

## 🚀 Usage

To simulate persona behavior:

```python
from persona_01_ratermax import simulate_ratermax_behavior
simulate_ratermax_behavior('path/to/input.csv', 'synthetic_data/products_ratermax.csv')
```

Or use the master script:

```python
from generate_all_personas import generate_all_personas
generate_all_personas('data/products.csv', 'synthetic_data/')
```

---

## 🛠 Branch: `syncdata` for Dataset Contribution

All synthetic datasets and persona logic reside in the dedicated Git branch: **`syncdata`**.

> ⚠️ **Precautions before pushing:**

1. Ensure your dataset files are cleaned:
   - No **null values** in any column
   - Drop `image` and `link` columns
2. File format must be **CSV** with consistent headers
3. Filename must include persona name, e.g., `products_ratermax.csv`
4. Do **not push to `main` or `dev`** directly

---

## ✅ CI/CD CSV Validation

The `syncdata` branch is protected with **GitHub Actions CI/CD validation**, which checks:

- ✅ No missing (`null`) values
- ✅ Required columns: `name`, `main_category`, `sub_category`, `rating`, `no_of_ratings`, `discount_price`, `actual_price`
- ✅ Valid persona naming in filenames
- ✅ Format consistency for production merge

Pull requests will be rejected automatically if validation fails.

---

## 📬 Contribution Flow

1. Create your persona logic
2. Generate CSVs
3. Push only to `syncdata`
4. Create a PR (pull request) to `dev` or `main`
5. Sathwik will review and merge after validation

---

## 👨‍💻 Author

Built by **Sathwik** – Federated Learning Model Architect & Team Lead  
Contributions, questions or improvements? Please open an issue or pull request.

---