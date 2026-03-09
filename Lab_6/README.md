# 🗂️ Lab 6 — Python Data Structures: Lists, Dictionaries & Loops

> **Course:** Artificial Intelligence (BS-CS)
> **Week:** 6
> **Author:** Muhammad Asif
> **Lab Type:** Hands-on Python / Jupyter Lab (Concepts + Coding)

---

## 📌 Overview

This lab provides a deep dive into Python's core **data structures** — lists and dictionaries — and demonstrates how they are used together with loops to manage and process structured data. These structures underpin almost every AI and data science application, from storing model outputs to managing datasets.

---

## 🎯 Learning Objectives

By the end of this lab, students will be able to:

- Create, modify, and access elements in **Python lists**
- Create, update, and query **Python dictionaries**
- Use `for` and `while` loops to **iterate** over data structures
- Apply data structure operations in a practical robotics-themed context

---

## 📂 Files

| File | Description |
|------|-------------|
| `AI_LAB6.ipynb` | Lab notebook — all concepts and tasks |
| `AI_Lab_Week_6.pdf` | Lab manual / reference sheet |

---

## 🧪 Concepts & Tasks Covered

### Concept 1 — Lists in Python
Create and manipulate an ordered collection of items.

```python
robots = ["Alpha", "Beta", "Gamma"]

print("Robot List:", robots)
print("First Robot:", robots[0])   # Positive index
print("Last Robot:",  robots[-1])  # Negative index

# Modify the list
robots.append("Delta")   # Add to end
robots.remove("Beta")    # Remove by value
print("Updated List:", robots)
```

**Key list operations:**

| Method | Description |
|--------|-------------|
| `append(x)` | Add element `x` to the end |
| `remove(x)` | Remove first occurrence of `x` |
| `len(list)` | Get the number of elements |
| `list[i]` | Access element at index `i` |

### Concept 2 — Dictionaries in Python
Store and manage key-value structured data.

```python
robot = {
    "name":    "Alpha",
    "battery": 80,
    "status":  "Active"
}

print("Robot Name:", robot["name"])
print("Battery:",    robot["battery"])

robot["battery"] = 100              # Update existing value
robot["task"]    = "Line Following" # Add new key-value pair
```

**Key dictionary operations:**

| Operation | Description |
|-----------|-------------|
| `dict[key]` | Access value by key |
| `dict[key] = value` | Set or update a value |
| `dict.keys()` | View all keys |
| `dict.values()` | View all values |
| `dict.items()` | Iterate key-value pairs |

### Concept 3 — Loops with Data Structures

**`for` loop** — iterate over a list:
```python
robots = ["Alpha", "Gamma", "Delta"]
for r in robots:
    print("Activating robot:", r)
```

**`while` loop** — decrement until condition fails:
```python
battery = 5
while battery > 0:
    print("Battery remaining:", battery)
    battery -= 1
print("Battery depleted!")
```

---

## 🛠️ How to Run

```bash
jupyter lab Lab_6/AI_LAB6.ipynb
```

Run each cell sequentially from top to bottom.

---

## 📚 Key Concepts

| Concept | Description |
|---------|-------------|
| List | Ordered, mutable sequence of elements |
| Dictionary | Unordered key-value store for structured data |
| Indexing | Access elements by position (`list[0]`, `list[-1]`) |
| `for` loop | Iterate over every element in a sequence |
| `while` loop | Repeat a block while a condition holds |
| Mutation | Modifying data structures in-place (`append`, `remove`) |

---

## 🔗 Related Labs

- **Lab 1** → Python Basics (variables, loops)
- **Lab 3** → Dictionaries & NLP Intro
- **Lab 7** → Python Modules & Libraries
