# 🏠 AI Lab 10 — Exploratory Data Analysis & Property Price Prediction

> **Course:** Artificial Intelligence (BS-CS)
> **Week:** 10+ (Extended Lab)
> **Author:** Muhammad Asif
> **Lab Type:** Machine Learning Project — EDA + Regression + Classification

---

## 📌 Overview

This lab applies **Exploratory Data Analysis (EDA)** and **supervised Machine Learning** to a real-world **property/housing dataset**. Students analyse the data visually and statistically, then train both a **Linear Regression** model and a **Random Forest Classifier** to extract insights and make predictions.

---

## 🎯 Learning Objectives

By the end of this lab, students will be able to:

- Load and inspect a real dataset using **Pandas**
- Perform **EDA** with descriptive statistics and visualisations (Matplotlib)
- Preprocess data using **LabelEncoder** for categorical features
- Split data into **train/test sets** for unbiased model evaluation
- Train and evaluate a **Linear Regression** model for continuous prediction
- Train and evaluate a **Random Forest Classifier** for categorical prediction
- Interpret **model performance metrics** (accuracy, R², MAE/MSE)

---

## 📂 Files

| File | Description |
|------|-------------|
| `Property Model.ipynb` | Full lab notebook — EDA through model evaluation |
| `Property.csv` | Property dataset *(required, not included — add your own)* |

---

## 🧪 Workflow

### Step 1 — Import Libraries

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error
```

### Step 2 — Load & Inspect Dataset

```python
df = pd.read_csv("Property.csv")

print("Shape:", df.shape)           # (rows, columns)
print("Columns:", df.columns.tolist())
print(df.head())                     # First 5 rows
print(df.describe())                 # Descriptive statistics
print(df.isnull().sum())             # Missing value counts
```

### Step 3 — EDA (Exploratory Data Analysis)

```python
# Distribution plot
df['price'].hist(bins=30, figsize=(8, 4))
plt.title("Property Price Distribution")
plt.xlabel("Price")
plt.show()

# Correlation heatmap
import seaborn as sns
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.show()
```

### Step 4 — Preprocessing

```python
le = LabelEncoder()
df['location_encoded'] = le.fit_transform(df['location'])

X = df[['area', 'bedrooms', 'bathrooms', 'location_encoded']]
y = df['price']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### Step 5 — Linear Regression (Price Prediction)

```python
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
print(f"MSE: {mse:.2f}")
print(f"R² Score: {lr.score(X_test, y_test):.4f}")
```

### Step 6 — Random Forest Classifier (Category Prediction)

```python
rfc = RandomForestClassifier(n_estimators=100, random_state=42)
rfc.fit(X_train, y_train_cat)
y_pred_cat = rfc.predict(X_test)

print(f"Accuracy: {accuracy_score(y_test_cat, y_pred_cat):.4f}")
```

---

## 🛠️ How to Run

1. Place `Property.csv` in the `AI_Lab10/` directory
2. Open and run the notebook:

```bash
jupyter lab AI_Lab10/"Property Model.ipynb"
```

---

## 📚 Key Concepts

| Concept | Description |
|---------|-------------|
| EDA | Statistical and visual exploration of a dataset |
| `DataFrame` | Pandas table structure for structured data |
| `LabelEncoder` | Convert categorical strings to numeric labels |
| Train/Test Split | Separate data for training and unbiased evaluation |
| Linear Regression | Predict a continuous output from input features |
| Random Forest | Ensemble of decision trees for classification |
| R² Score | Coefficient of determination (1.0 = perfect fit) |
| MSE | Mean Squared Error — lower is better |
| Accuracy | Fraction of correctly classified samples |

---

## 🔗 Related Labs

- **Lab 9** → NumPy Arrays (data manipulation foundation)
- **Lab 14/15** → Model Representation, Cost Function & Gradient Descent
