# 🏋️ Fitness Dataset Augmentation & Balancing Project

## 📖 Overview
This notebook focuses on **expanding and balancing a fitness dataset** that originally contained **973 rows and 44 columns**.  
The goal was to **generate realistic synthetic data** and increase the dataset size to **20,000 rows**, while maintaining logical consistency between related features.

---

## ⚙️ Steps Summary

### 1. 📥 Data Loading & Inspection
- Loaded the original dataset.
- Checked for **missing values**, **data types**, and **overall shape**.
- Performed **initial exploratory analysis** on exercise types, calories, and nutrient columns.

### 2. 🧩 Missing Value Handling
- Filled missing values using:
  - Statistical measures (`mean`, `median`) for numerical columns.
  - Random selection or mode for categorical columns.
- Ensured `rating`, `sugar_g`, and other nutritional columns had **no nulls** remaining.

### 3. 📈 Data Balancing & Expansion
- The original data was **unbalanced** (some exercises and meals were overrepresented).
- We used a **smart duplication and sampling method** with randomization to expand data:
  ```python
  df_large = pd.concat([df] * (target_rows // current_rows + 1), ignore_index=True)
  df_large = df_large.sample(n=target_rows, replace=True, random_state=42).reset_index(drop=True)
