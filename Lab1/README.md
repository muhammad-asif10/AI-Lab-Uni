# 🐍 Lab 1 — Python Fundamentals for AI

> **Course:** Artificial Intelligence (BS-CS)
> **Week:** 1 & 2
> **Author:** Muhammad Asif
> **Lab Type:** Hands-on Python / Jupyter Lab (Concepts + Coding)

---

## 📌 Overview

This lab introduces the foundational Python programming concepts required for Artificial Intelligence development. Students learn how to write clean, structured Python code including variables, control flow, loops, and functions — the building blocks for every AI program.

---

## 🎯 Learning Objectives

By the end of this lab, students will be able to:

- Declare and use **variables** with different data types (`int`, `float`, `str`)
- Apply **conditional logic** (`if / elif / else`) to solve decision-making problems
- Use **loops** (`for`, `while`) for repetitive tasks and data iteration
- Write reusable **functions** to modularise code
- Implement basic **input validation** and **authentication** logic

---

## 📂 Files

| File | Description |
|------|-------------|
| `AI_Lab.ipynb/AI_Lab1_Asif.ipynb` | Core lab notebook — variables, loops, functions *(stored inside a folder named `AI_Lab.ipynb`)* |
| `AI_lab2_Asif.ipynb` | Extended tasks — conditionals, comparisons, grade checker, login system |
| `AI Lab Week 1.pdf` | Lab manual / reference sheet |

---

## 🧪 Tasks Covered

### Task 1 — Variables & Data Types
Store and display student information using Python variables (`str`, `int`, `float`).

```python
name = "Ali"
roll_no = 23
cgpa = 3.4
print("Name:", name)
```

### Task 2 — Loops
Iterate and print sequences using `for` and `while` loops.

```python
for i in range(1, 11):
    print(i)
```

### Task 3 — Functions
Write a reusable function to calculate the **factorial** of a number.

```python
def factorial(n):
    result = 1
    for i in range(1, n + 1):
        result *= i
    return result
```

### Task 4 — Conditionals & Comparison
Compare two numbers and determine the greater value; assign letter grades based on marks.

### Task 5 — Even / Odd Checker
Determine whether a user-provided number is even or odd using the modulo operator.

### Task 6 — Login System (While Loop + Input Validation)
Simulate a password-based login system with a limited number of attempts.

```python
CORRECT_USER = "admin"
CORRECT_PASS = "1234"
attempts = 3

while attempts > 0:
    u = input("Username: ")
    p = input("Password: ")
    if u == CORRECT_USER and p == CORRECT_PASS:
        print("Access granted!")
        break
    attempts -= 1
```

---

## 🛠️ How to Run

1. Open the notebook in **Jupyter Lab** or **Google Colab**
2. Run cells sequentially from top to bottom
3. Provide input when prompted by `input()` calls

```bash
# Note: AI_Lab.ipynb is a folder — navigate into it
jupyter lab "Lab1/AI_Lab.ipynb/AI_Lab1_Asif.ipynb"
```

---

## 📚 Key Concepts

| Concept | Description |
|---------|-------------|
| Variables | Named containers for storing data values |
| `if / elif / else` | Branching logic based on conditions |
| `for` loop | Iterate over a range or sequence |
| `while` loop | Repeat until a condition becomes `False` |
| Functions (`def`) | Reusable, parameterised code blocks |
| `input()` | Read user input from the console |

---

## 🔗 Related Labs

- **Lab 3** → Python Dictionaries & NLP Intro
- **Lab 4** → Functions, NumPy & Text Preprocessing
