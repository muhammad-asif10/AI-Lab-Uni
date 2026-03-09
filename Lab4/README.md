# 🔢 Lab 4 — Functions, NumPy Basics & NLP Text Preprocessing

> **Course:** Artificial Intelligence (BS-CS Semester 3)
> **Week:** 4
> **Author:** Muhammad Asif
> **Lab Type:** Hands-on Python / Jupyter Lab (Concepts + Coding)

---

## 📌 Overview

This lab bridges **core Python programming** with foundational **AI and data science** tools. Students practise writing reusable functions, apply simple NLP text-cleaning techniques, and explore **NumPy** for numerical computation — skills directly applicable to building AI models.

---

## 🎯 Learning Objectives

By the end of this lab, students will be able to:

- Write and call **functions** with parameters and return values
- Perform **NLP text preprocessing** (lowercasing, punctuation removal, tokenisation)
- Create and analyse **NumPy arrays** (sum, mean, max, min, boolean masking)
- Build a **rule-based decision system** using conditional logic

---

## 📂 Files

| File | Description |
|------|-------------|
| `AI_Lab_Week4.ipynb` | Lab notebook — all four tasks |
| `AI_Lab_Week_4.pdf` | Lab manual / reference sheet |

---

## 🧪 Tasks Covered

### Task 1 — Student Grading Function
Write a reusable function `get_grade(marks)` that maps numeric scores to letter grades.

```python
def get_grade(marks):
    """Return a letter grade based on numeric marks."""
    if marks >= 80:
        return "A"
    elif marks >= 60:
        return "B"
    elif marks >= 40:
        return "C"
    else:
        return "Fail"

marks_list = [95, 72, 38, 60, 45]
for m in marks_list:
    print(f"Marks: {m} → Grade: {get_grade(m)}")
```

### Task 2 — NLP Text Cleaner
Clean raw text as a preprocessing step for AI language models: lowercase, strip punctuation, and tokenise.

```python
import string

sentence = "Hello, Students! AI is amazing, isn't it?"
sentence_lower = sentence.lower()
cleaned = sentence_lower
for p in string.punctuation:
    cleaned = cleaned.replace(p, "")
words = cleaned.split()
print("Cleaned tokens:", words)
```

> **Why it matters:** Text cleaning is the first step in any NLP pipeline (sentiment analysis, text classification, chatbots).

### Task 3 — NumPy Array Statistics
Create a NumPy array and compute descriptive statistics, then filter values above the mean using boolean masking.

```python
import numpy as np

arr = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
print("Sum:", arr.sum())
print("Mean:", arr.mean())
print("Max:", arr.max(), "| Min:", arr.min())

above_mean = arr[arr > arr.mean()]
print("Above mean:", above_mean)
```

### Task 4 — Mini Rule-Based Health Advisor
Build a small rule-based AI that suggests health tips based on how the user reports feeling.

---

## 🛠️ How to Run

```bash
jupyter lab Lab4/AI_Lab_Week4.ipynb
```

Run each cell sequentially from top to bottom.

---

## 📚 Key Concepts

| Concept | Description |
|---------|-------------|
| Functions (`def`) | Reusable, parameterised code blocks |
| NLP Preprocessing | Lowercasing, punctuation removal, tokenisation |
| NumPy `ndarray` | Fast, memory-efficient numerical arrays |
| Boolean Masking | Filter array elements using True/False conditions |
| Descriptive Stats | Sum, mean, max, min on arrays |
| Rule-Based Systems | Conditional logic to simulate AI decisions |

---

## 🔗 Related Labs

- **Lab 3** → Python Dictionaries & NLP Intro
- **Lab 5** → Rule-Based AI Systems
- **Lab 9** → NumPy Arrays Deep Dive
