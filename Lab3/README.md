# 📖 Lab 3 — Python Dictionaries & NLP Introduction

> **Course:** Artificial Intelligence (BS-CS)
> **Week:** 3
> **Author:** Muhammad Asif
> **Lab Type:** Hands-on Python / Jupyter Lab (Concepts + Coding)

---

## 📌 Overview

This lab introduces **Python dictionaries** as a fundamental data structure and demonstrates their direct relevance to **Natural Language Processing (NLP)**. Through four progressive tasks, students build real-world mini-systems including a word counter, a rule-based chatbot, and a movie recommender — all powered by dictionaries.

---

## 🎯 Learning Objectives

By the end of this lab, students will be able to:

- Create and manipulate **Python dictionaries** (key-value pairs)
- Iterate over dictionaries using `.items()`, `.keys()`, and `.values()`
- Build a **word frequency counter** — a core NLP pre-processing technique
- Implement a **rule-based chatbot** using dictionary-mapped responses
- Design a **basic recommender system** using the `random` library

---

## 📂 Files

| File | Description |
|------|-------------|
| `AI_Lab3_Asif.ipynb` | Lab notebook — all four tasks |
| `AI Lab Week 3.pdf` | Lab manual / reference sheet |

---

## 🧪 Tasks Covered

### Task 1 — Student Marks & Grades
Store student names and marks in a dictionary and use a function with `if/elif/else` to assign letter grades.

```python
students = {"Ali": 85, "Sara": 67, "Bilal": 45}

def print_results(data):
    for name, marks in data.items():
        if marks >= 80:
            grade = "A"
        elif marks >= 60:
            grade = "B"
        else:
            grade = "C"
        print(f"{name}: {marks} → Grade {grade}")
```

### Task 2 — Word Counter (NLP Pre-processing)
Take a sentence from the user and use a dictionary to count word occurrences — a core step in NLP pipelines.

```python
sentence = input("Enter a sentence: ")
words = sentence.split()
counter = {}
for w in words:
    w = w.lower()
    counter[w] = counter.get(w, 0) + 1
print("Word frequencies:", counter)
```

### Task 3 — Simple Rule-Based Chatbot
Build a minimal chatbot that maps user phrases to predefined responses using a dictionary lookup.

```python
responses = {
    "hi": "Hello! How can I help you?",
    "bye": "Goodbye! Have a nice day!",
    "how are you": "I'm doing great, thanks!"
}
user = input("You: ").lower()
print("Bot:", responses.get(user, "Sorry, I don't understand."))
```

### Task 4 — Movie Recommender
Simulate a genre-based movie recommender that randomly selects a film from a dictionary of genre-to-movie-list mappings.

```python
import random
movies = {
    "action": ["Avengers", "Batman", "Spiderman"],
    "comedy": ["Mr. Bean", "Home Alone", "The Mask"],
    "sci-fi": ["Interstellar", "Inception", "The Matrix"]
}
genre = input("Enter genre (action/comedy/sci-fi): ").lower()
print("Recommended:", random.choice(movies.get(genre, ["No movies found"])))
```

---

## 🛠️ How to Run

```bash
jupyter lab Lab3/AI_Lab3_Asif.ipynb
```

Run each cell sequentially and provide input when prompted.

---

## 📚 Key Concepts

| Concept | Description |
|---------|-------------|
| Dictionary | Key-value data structure for fast lookups |
| `.items()` | Iterate over key-value pairs |
| `.get(key, default)` | Safe dictionary access with a fallback value |
| Word Frequency | Counting token occurrences — foundational NLP step |
| Rule-Based Chatbot | Response selection via dictionary matching |
| `random.choice()` | Pick a random element from a list |

---

## 🔗 Related Labs

- **Lab 1** → Python Basics (variables, loops, functions)
- **Lab 4** → Functions, NumPy & NLP Text Preprocessing
- **Lab 5** → Rule-Based AI Systems
