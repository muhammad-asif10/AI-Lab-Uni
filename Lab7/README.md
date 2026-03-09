# 📦 Lab 7 — Python Modules: Built-in, Custom & Third-Party

> **Course:** Artificial Intelligence (BS-CS)
> **Week:** 7 / 8
> **Author:** Muhammad Asif
> **Lab Type:** Hands-on Python / Jupyter Lab (Concepts + Coding)

---

## 📌 Overview

This lab covers **Python modules and libraries** — the mechanism that allows AI developers to reuse, share, and extend functionality. Students practise creating their own modules, using Python's built-in standard library, and importing third-party packages like NumPy, `requests`, and `random`.

---

## 🎯 Learning Objectives

By the end of this lab, students will be able to:

- Create a **custom Python module** and import it in another script
- Use **built-in modules** (`math`, `datetime`, `time`, `random`)
- Import and use **third-party libraries** (`numpy`, `requests`)
- Understand **selective imports** (`from module import function`)

---

## 📂 Files

| File | Description |
|------|-------------|
| `AI_LAB7.ipynb` | Main lab notebook |
| `mymodule.py` | Custom module with `greet()`, `add_numbers()`, `multiply_numbers()` |
| `main.py` | Script demonstrating module imports |
| `math_Time.py` | Built-in `math` and `datetime` usage |
| `numpy.py` | NumPy operations example |
| `random.py` | `random` module usage |
| `requests.py` | HTTP requests with the `requests` library |
| `random_number.txt` | Output file from random number generation |
| `Week07_Builtins_ThirdParty_Lab.pdf` | Lab manual / reference sheet |

---

## 🧪 Concepts & Tasks Covered

### Concept 1 — Custom Module (`mymodule.py`)
Create a reusable module with documented functions.

```python
# mymodule.py
def greet(name):
    """A simple function to greet someone."""
    return f"Hello, {name}! Welcome to Week 8."

def add_numbers(a, b):
    """Adds two numbers and returns the result."""
    return a + b

def multiply_numbers(a, b):
    """Multiplies two numbers and returns the result."""
    return a * b
```

### Concept 2 — Import Styles

**Import entire module:**
```python
import mymodule
print(mymodule.greet("Student"))
print(mymodule.add_numbers(5, 10))
```

**Selective import (best practice):**
```python
from mymodule import greet, add_numbers
print(greet("Student"))
print(add_numbers(5, 10))
```

### Concept 3 — Built-in Modules: `math` & `datetime`

```python
import math
from datetime import datetime

# Mathematical operations
print(f"Square root of 16: {math.sqrt(16)}")       # 4.0
print(f"Pi: {math.pi:.4f}")                         # 3.1416
print(f"Circle area (r=5): {math.pi * 5**2:.2f}")  # 78.54

# Date and time
now = datetime.now()
print(f"Current date/time: {now.strftime('%Y-%m-%d %H:%M:%S')}")
```

### Concept 4 — Third-Party Library: NumPy

```python
import numpy as np

arr = np.array([1, 2, 3, 4, 5])
print("Array:", arr)
print("Mean:", arr.mean())
print("Sum:", arr.sum())
```

### Concept 5 — HTTP Requests Library

```python
import requests

response = requests.get("https://api.github.com")
print("Status Code:", response.status_code)
```

---

## 🛠️ How to Run

**Run the notebook:**
```bash
jupyter lab Lab7/AI_LAB7.ipynb
```

**Run individual scripts:**
```bash
python Lab7/main.py
python Lab7/math_Time.py
python Lab7/numpy.py
```

---

## 📚 Key Concepts

| Concept | Description |
|---------|-------------|
| Module | A `.py` file containing reusable functions and variables |
| `import` | Load a module into the current namespace |
| `from … import` | Import only specific names from a module |
| Standard Library | Built-in Python modules (no install needed) |
| Third-Party Library | Packages installed via `pip` (numpy, requests) |
| `math` module | Mathematical functions and constants |
| `datetime` module | Date and time manipulation |
| `random` module | Pseudo-random number generation |
| `numpy` | Fast numerical array computation |
| `requests` | HTTP client for web APIs |

---

## 🔗 Related Labs

- **Lab 6** → Python Data Structures
- **Lab 9** → NumPy Arrays Deep Dive
- **Lab 10** → EDA & Machine Learning with scikit-learn
