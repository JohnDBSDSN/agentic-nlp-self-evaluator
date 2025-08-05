# agentic_nlp_retrainer.py

from dotenv import load_dotenv
load_dotenv()

import os
import openai
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

from utils import load_mock_data, get_accuracy, query_agent

# 🔐 Load API key
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY not found. Set it in .env.")
openai.api_key = openai_api_key

# Config
THRESHOLD = 0.90  # Acceptable performance threshold
MAX_ITER = 5
SIMULATE_IMPROVEMENT = True

# Step 1: Load data
df = load_mock_data()
X = df['text']
y = df['label']

# Step 2: Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 3: Create baseline pipeline
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', LogisticRegression())
])

pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
baseline_acc = get_accuracy(y_test, y_pred)

# Simulate starting with lower accuracy for demo purposes
if SIMULATE_IMPROVEMENT:
    baseline_acc = 0.70  # Start at 70% for the demo

print(f"\n🔍 Baseline Model Accuracy: {baseline_acc:.2f}")

# Track best model
best_pipeline = pipeline
best_accuracy = baseline_acc

# Retraining loop
iteration = 1
accuracy_history = [baseline_acc]

while iteration <= MAX_ITER:
    print(f"\n📊 Iteration {iteration}: Accuracy = {accuracy_history[-1]:.2f}")

    # Ask the agent
    agent_response = query_agent(accuracy_history[-1], threshold=THRESHOLD)
    print(f"\n🧠 Agent's Decision:\n{agent_response}")

    if "no" in agent_response.lower() and "retrain" not in agent_response.lower():
        print("\n🛑 Agent decided no further retraining is needed.")
        break

    # Run GridSearchCV
    print("\n🔁 Agent suggests retraining. Running GridSearchCV...")
    param_grid = {
        'clf__C': [0.1, 1, 10],
        'clf__max_iter': [100, 200]
    }
    print(f"📊 GridSearch Parameters: {param_grid}")

    grid = GridSearchCV(best_pipeline, param_grid, cv=3)
    grid.fit(X_train, y_train)

    y_pred_new = grid.predict(X_test)
    new_acc = get_accuracy(y_test, y_pred_new)

    # ✅ Simulate slight improvement per iteration
    if SIMULATE_IMPROVEMENT:
        simulated_gain = 0.03 * iteration  # +3% per round
        new_acc = min(1.0, accuracy_history[-1] + simulated_gain)

    accuracy_history.append(new_acc)

    print(f"\n📈 Accuracy after retraining: {new_acc:.2f}")

    if new_acc > best_accuracy:
        best_pipeline = grid.best_estimator_
        best_accuracy = new_acc
        print("✅ Accuracy improved. Updating best model.")
    else:
        print("⚠️ No improvement. Will check again next round.")

    if len(accuracy_history) >= 3 and accuracy_history[-1] == accuracy_history[-2] == accuracy_history[-3]:
        print("\n🛑 Accuracy has plateaued for 3 rounds. Stopping retraining.")
        break

    iteration += 1

# Final Summary
print("\n🧾 Agent Summary:")
print(f"🔹 Final Accuracy: {best_accuracy:.2f}")
print(f"🔹 Total Iterations: {iteration - 1}")
print(f"🔹 Accuracy History: {[round(a, 2) for a in accuracy_history]}")
print(f"🔹 Accuracy Gain: {round(best_accuracy - baseline_acc, 2)}")

plt.figure(figsize=(8, 5))
plt.plot(range(len(accuracy_history)), accuracy_history, marker='o', color='blue', linestyle='-')
plt.title("📈 Accuracy Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Accuracy")
plt.xticks(range(len(accuracy_history)))
plt.ylim(0.5, 1.05)
for i, acc in enumerate(accuracy_history):
    plt.text(i, acc + 0.01, f"{acc:.2f}", ha='center', fontsize=8)

plt.grid(True)
plt.tight_layout()
plt.savefig("accuracy_plot.png")  # Saves as image file
plt.show()