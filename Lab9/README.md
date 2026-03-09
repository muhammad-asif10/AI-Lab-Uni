# 🔢 Lab 9 — NumPy Arrays: Indexing, Slicing & Operations

> **Course:** Artificial Intelligence (BS-CS — Evening)
> **Week:** 10
> **Author:** Muhammad Asif | Roll No: 2024-CSRE-008
> **Instructor:** Mr. Azeem Aslam
> **Lab Type:** Hands-on Python / Jupyter Lab (Concepts + Coding)

---

## 📌 Overview

This lab provides a comprehensive introduction to **NumPy** — the foundational numerical computing library for Python. NumPy arrays (`ndarray`) are the core data structure used in virtually every AI, machine learning, and data science framework. Students explore array creation, indexing, slicing, reshaping, and vectorised operations.

---

## 🎯 Learning Objectives

By the end of this lab, students will be able to:

- Explain what **NumPy** is and why it is essential for AI
- Create **1D and 2D arrays** using various NumPy functions
- Apply **indexing and slicing** to extract elements and sub-arrays
- Perform **element-wise and aggregate operations** on arrays
- Understand the performance advantage of NumPy over Python lists

---

## 📂 Files

| File | Description |
|------|-------------|
| `AI_LAB9.ipynb` | Lab notebook — NumPy concepts and coding exercises |

---

## 🧪 Concepts & Tasks Covered

### Part 1 — Why NumPy?

In AI we work with large volumes of numbers (image pixels, sensor readings, model weights). Python lists are slow for this. NumPy provides:

- A specialised `ndarray` data type
- Vectorised operations (no explicit loops needed)
- Broadcasting for operations on arrays of different shapes
- A bridge to TensorFlow, PyTorch, scikit-learn, and Pandas

```python
import numpy as np
print("NumPy version:", np.__version__)
```

### Part 2 — Creating NumPy Arrays

```python
# From a Python list
arr1d = np.array([10, 20, 30, 40, 50])

# 2D array (matrix)
arr2d = np.array([[1, 2, 3],
                  [4, 5, 6]])

# Convenience constructors
zeros   = np.zeros((3, 3))        # 3×3 matrix of zeros
ones    = np.ones((2, 4))         # 2×4 matrix of ones
rng     = np.arange(0, 10, 2)    # [0, 2, 4, 6, 8]
linsp   = np.linspace(0, 1, 5)   # 5 evenly spaced values
```

### Part 3 — Indexing & Slicing

```python
arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

# Single element
print(arr[0])    # 10
print(arr[-1])   # 100

# Slice
print(arr[2:6])  # [30, 40, 50, 60]

# 2D array indexing
mat = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(mat[1, 2])     # 6  (row 1, col 2)
print(mat[:, 1])     # [2, 5, 8]  (all rows, col 1)
print(mat[0:2, :])   # First two rows
```

### Part 4 — Array Operations

```python
a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])

# Element-wise arithmetic
print(a + b)    # [11, 22, 33, 44, 55]
print(a * b)    # [ 10,  40,  90, 160, 250]
print(b / a)    # [10., 10., 10., 10., 10.]

# Aggregate statistics
print("Sum:",  a.sum())
print("Mean:", a.mean())
print("Max:",  a.max())
print("Min:",  a.min())
print("Std:",  a.std())
```

### Part 5 — Boolean Masking

```python
arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
mask = arr > arr.mean()           # Boolean array
print("Above mean:", arr[mask])  # [60, 70, 80, 90, 100]
```

### Part 6 — Reshape & Transpose

```python
arr = np.arange(1, 13)          # [1, 2, ..., 12]
matrix = arr.reshape(3, 4)      # 3 rows × 4 cols
print(matrix)

print("Transposed:\n", matrix.T) # 4 rows × 3 cols
```

---

## 🛠️ How to Run

```bash
jupyter lab Lab9/AI_LAB9.ipynb
```

---

## 📚 Key Concepts

| Concept | Description |
|---------|-------------|
| `ndarray` | NumPy's core n-dimensional array type |
| `np.array()` | Create array from a Python list |
| `np.zeros()` / `np.ones()` | Arrays pre-filled with 0s or 1s |
| `np.arange()` | Range-based array creation |
| `np.linspace()` | Evenly spaced values within an interval |
| Slicing `[start:stop]` | Extract a sub-array |
| Boolean masking | Filter elements using a True/False condition |
| `.reshape()` | Change array dimensions without copying data |
| `.T` | Transpose a matrix |
| Vectorised ops | Element-wise arithmetic without explicit loops |

---

## 🔗 Related Labs

- **Lab 4** → First encounter with NumPy basics
- **Lab 7** → Importing NumPy as a third-party library
- **Lab 10** → Applying NumPy in EDA & Machine Learning pipelines
