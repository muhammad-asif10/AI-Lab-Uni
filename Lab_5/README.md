# 🤖 Lab 5 — Rule-Based AI Systems

> **Course:** Artificial Intelligence (BS-CS)
> **Week:** 5
> **Author:** Muhammad Asif
> **Lab Type:** Hands-on Python / Jupyter Lab (Concepts + Coding)

---

## 📌 Overview

This lab explores **rule-based Artificial Intelligence** — one of the earliest and most interpretable paradigms in AI. Students implement four distinct systems: an **intent classifier**, an **expert system**, a **weighted recommender**, and a **Finite State Machine (FSM)** — all without any machine learning model, using pure logic and structured rules.

---

## 🎯 Learning Objectives

By the end of this lab, students will be able to:

- Build a **keyword-based intent classifier** to detect user intentions
- Design an **expert system** that applies domain rules to make decisions
- Implement a **weighted scoring recommender** that ranks options by multiple criteria
- Model a **Finite State Machine (FSM)** to represent sequential state transitions

---

## 📂 Files

| File | Description |
|------|-------------|
| `AI_Lab5.ipynb` | Lab notebook — all four tasks |
| `AI_Lab_Week_5.pdf` | Lab manual / reference sheet |

---

## 🧪 Tasks Covered

### Task 1 — Mini Intent Classifier (Keyword Rules)
Map user input to an intent category by scanning for matching keywords in a dictionary.

```python
intent_keywords = {
    "greeting": ["hi", "hello", "hey", "salam"],
    "goodbye":  ["bye", "see you", "take care"],
    "thanks":   ["thanks", "thank you"],
    "food":     ["eat", "restaurant", "food", "lunch", "dinner"],
    "study":    ["study", "exam", "assignment", "quiz"],
}

def classify_intent(user_input):
    user_input = user_input.lower()
    for intent, keywords in intent_keywords.items():
        if any(kw in user_input for kw in keywords):
            return intent
    return "unknown"
```

> **AI Connection:** Intent classification is the first layer of every virtual assistant (Siri, Alexa, ChatGPT).

### Task 2 — Expert System: Scholarship Eligibility
Encode domain knowledge as explicit rules to determine scholarship eligibility.

```python
def decide_scholarship(cgpa, family_income, achievements):
    if cgpa >= 3.7 and family_income <= 60000:
        return "Full Scholarship"
    elif cgpa >= 3.0 and family_income <= 120000:
        return "Partial Scholarship"
    else:
        return "Not Eligible"
```

> **AI Connection:** Expert systems encode human expertise into if-then rules — a cornerstone of early AI (MYCIN, DENDRAL).

### Task 3 — Rule-Based Recommender (Weighted Scoring)
Score multiple activity options based on the user's current feeling and time of day, then recommend the highest-scoring option.

```python
def activity_recommender(feeling, time_of_day):
    scores = {"read": 0, "walk": 0, "nap": 0, "music": 0}
    if feeling == "tired":
        scores["nap"] += 3
        scores["music"] += 1
    if time_of_day == "morning":
        scores["walk"] += 2
        scores["read"] += 1
    return max(scores, key=scores.get)
```

> **AI Connection:** Weighted scoring is the foundation of collaborative filtering and content-based recommendation systems.

### Task 4 — Finite State Machine (FSM): Traffic Light
Model a traffic light as an FSM where each state has a defined next state and duration.

```python
fsm = {
    "RED":    {"next": "GREEN",  "duration": 1},
    "GREEN":  {"next": "YELLOW", "duration": 1},
    "YELLOW": {"next": "RED",    "duration": 1},
}

def run_traffic_light(cycles=3):
    state = "RED"
    for _ in range(cycles * len(fsm)):
        print(f"State: {state}")
        state = fsm[state]["next"]
```

> **AI Connection:** FSMs model agent behaviour, dialogue systems, and game AI state transitions.

---

## 🛠️ How to Run

```bash
jupyter lab Lab_5/AI_Lab5.ipynb
```

Run each cell sequentially from top to bottom.

---

## 📚 Key Concepts

| Concept | Description |
|---------|-------------|
| Intent Classification | Detecting user intent from text using keyword matching |
| Expert System | Rule-encoded domain knowledge for automated decisions |
| Weighted Scoring | Ranking options by assigning numeric scores per criterion |
| Finite State Machine | Model with discrete states and defined transitions |
| Rule-Based AI | AI decisions driven purely by human-defined logic |

---

## 🔗 Related Labs

- **Lab 3** → Dictionaries & Rule-Based Chatbot
- **Lab 4** → Functions & NLP Preprocessing
- **Lab 14/15** → From Rules to Learning: Model Representation & Cost Functions
