# 📉 Lab 14 & 15 — Model Representation, Cost Function & Gradient Descent

> **Course:** Artificial Intelligence (BS-CS)
> **Week:** 14 & 15
> **Author:** Muhammad Asif
> **Lab Type:** Mathematical Foundations + Coding (OOP + Gradient Descent)

---

## 📌 Overview

These two combined labs form the **mathematical heart of Machine Learning**. Students move from conceptual understanding to Python implementation — building a linear model from scratch, defining a cost function to measure prediction error, and implementing **Gradient Descent** to minimise that error iteratively.

---

## 🎯 Learning Objectives

By the end of this lab, students will be able to:

- Explain the **Linear Model** equation (`y = w·x + b`) in plain language
- Implement a `LinearModel` class in Python using OOP
- Define and compute the **Mean Squared Error (MSE) cost function**
- Understand the intuition behind **Gradient Descent**
- Implement a manual gradient descent loop to find optimal model parameters
- Visualise the training process (loss curve, prediction line)

---

## 📂 Files

| File | Description |
|------|-------------|
| `AI Lab 14_15.ipynb` | Main lab notebook — model, cost function, gradient descent |
| `AI_Lab_Week14_15_ExtraSimple_ModelCost_GD_UVAS_AILab.html` | Exported HTML version for easy viewing |

---

## 🧪 Concepts & Tasks Covered

### Concept 1 — Real-Life Analogy

| ML Term | Analogy |
|---------|---------|
| **Model** | Your throwing technique |
| **Prediction** | Where the ball lands |
| **Cost / Error** | Distance from the target |
| **Training** | Practising to reduce that distance |

---

### Concept 2 — Model Representation

The linear model equation:

```
y = w × x + b
```

| Symbol | Meaning |
|--------|---------|
| `x` | Input feature (e.g., house size in sq ft) |
| `w` | Weight / slope — how strongly `x` affects `y` |
| `b` | Bias / intercept — baseline value when `x = 0` |
| `y` | Predicted output (e.g., house price) |

**Python (OOP):**
```python
class LinearModel:
    def __init__(self, w, b):
        self.w = w   # weight
        self.b = b   # bias

    def predict(self, x):
        return self.w * x + self.b

# Example
model = LinearModel(w=2.5, b=10)
print(model.predict(100))  # 260
```

---

### Concept 3 — Cost Function (Mean Squared Error)

The cost function measures how wrong our model is. A lower cost means better predictions.

```
MSE = (1/N) × Σ (y_predicted - y_actual)²
```

```python
import numpy as np

def mean_squared_error(y_pred, y_true):
    errors = y_pred - y_true
    return np.mean(errors ** 2)

y_true = np.array([100, 200, 300, 400, 500])
y_pred = np.array([110, 190, 310, 390, 510])

cost = mean_squared_error(y_pred, y_true)
print(f"MSE Cost: {cost:.2f}")  # 100.0
```

---

### Concept 4 — Gradient Descent

Gradient Descent is an optimisation algorithm that iteratively updates `w` and `b` to **minimise the cost function**.

**Update Rule:**
```
w = w - α × (∂Cost/∂w)
b = b - α × (∂Cost/∂b)
```

- `α` (alpha) = **learning rate** (controls step size)
- Derivatives tell us the direction and magnitude to adjust parameters

**Python Implementation:**
```python
def gradient_descent(X, y, w=0.0, b=0.0, lr=0.01, epochs=1000):
    N = len(X)
    cost_history = []

    for epoch in range(epochs):
        y_pred = w * X + b
        error  = y_pred - y

        # Compute gradients
        dw = (2 / N) * np.dot(error, X)
        db = (2 / N) * np.sum(error)

        # Update parameters
        w -= lr * dw
        b -= lr * db

        cost = np.mean(error ** 2)
        cost_history.append(cost)

    return w, b, cost_history

X = np.array([1, 2, 3, 4, 5], dtype=float)
y = np.array([2, 4, 6, 8, 10], dtype=float)

w_opt, b_opt, history = gradient_descent(X, y, lr=0.01, epochs=500)
print(f"Optimal w={w_opt:.4f}, b={b_opt:.4f}")  # w≈2.0, b≈0.0
```

---

### Concept 5 — Visualising Training

```python
import matplotlib.pyplot as plt

# Loss curve
plt.plot(history)
plt.title("Training Loss (MSE) over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Cost (MSE)")
plt.show()

# Prediction line vs actual data
plt.scatter(X, y, label="Actual", color="blue")
plt.plot(X, w_opt * X + b_opt, label="Predicted", color="red")
plt.title("Linear Regression Fit")
plt.legend()
plt.show()
```

---

## 🛠️ How to Run

```bash
jupyter lab "Lab 14_15/AI Lab 14_15.ipynb"
```

Or open the pre-rendered HTML file in any browser:
```bash
open "Lab 14_15/AI_Lab_Week14_15_ExtraSimple_ModelCost_GD_UVAS_AILab.html"
```

---

## 📚 Key Concepts

| Concept | Description |
|---------|-------------|
| Linear Model `y = wx + b` | Simplest parametric model |
| Weight `w` | Slope — sensitivity of output to input |
| Bias `b` | Intercept — baseline output value |
| MSE | Mean Squared Error — standard regression loss |
| Gradient | Partial derivative indicating direction of steepest ascent |
| Learning Rate `α` | Step size for parameter updates |
| Gradient Descent | Iterative optimisation to minimise loss |
| Epoch | One full pass through the training data |
| Loss Curve | Plot of cost vs. epoch to monitor convergence |

---

## 🔗 Related Labs

- **Lab 10** → Applying Linear Regression with scikit-learn
- **Lab 9** → NumPy for vectorised gradient computations
- **Lab 5** → Rule-Based AI (contrast with learned models)
