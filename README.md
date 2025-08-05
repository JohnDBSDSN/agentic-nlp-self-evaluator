# 🤖 Agentic NLP Self-Evaluator

This project is a **beginner-friendly Agentic AI demo** that uses an LLM agent (OpenAI GPT) to evaluate the performance of a basic NLP model — and **decide whether to retrain it** based on accuracy. It combines:

- ✅ A simple **text classification** pipeline (Scikit-learn)
- ✅ Autonomous decision-making using **LLM as an agent**
- ✅ Automatic **fine-tuning via GridSearchCV**
- ✅ Iterative feedback loop with accuracy tracking
- ✅ Visualization of model improvement

---

## 📌 Key Concepts

- **Agentic AI**: The LLM plays the role of an evaluator agent, analyzing current model accuracy and deciding on next steps.
- **NLP Task**: Binary sentiment classification (positive/negative).
- **Self-evaluation loop**: GPT determines if retraining is needed based on thresholds.
- **Demonstration ready**: Ideal for LinkedIn, Medium, or interview portfolios.

---

## 🧠 How It Works

1. Load a labeled dataset of short text reviews.
2. Train a `CountVectorizer` + `LogisticRegression` pipeline.
3. Measure initial accuracy on test data.
4. Ask GPT (via OpenAI API) whether the model should be retrained.
5. If yes → run `GridSearchCV` and repeat.
6. Track improvements until the agent is satisfied or a limit is hit.
7. Plot and log the entire journey.

---

## 🏁 Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/JohnDBSDSN/agentic-nlp-self-evaluator.git
cd agentic-nlp-self-evaluator

2. Add your OpenAI API key

OPENAI_API_KEY=sk-...
❗ Never share this key — it’s already excluded via .gitignore.

3. Install dependencies
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt

4.Run the model
python agentic_nlp_retrainer.py

📁 Project Structure
agentic-nlp-self-evaluator/
│
├── agentic_nlp_retrainer.py       # Main logic + agent loop
├── utils.py                       # Support functions (load, accuracy, GPT query)
├── generate_better_reviews.py     # Creates synthetic input_reviews.csv
│
├── Data/
│   └── input_reviews.csv          # 100 labeled text samples (pos/neg)
│
├── .env                           # Your OpenAI API key (excluded from Git)
├── .gitignore
├── requirements.txt
└── README.md

📊 Sample Output
🔍 Baseline Model Accuracy: 0.70

🧠 Agent's Decision:
Yes, accuracy is low. Please retrain.

🔁 Running GridSearch...
✅ Accuracy improved to 0.79

🧠 Agent's Decision:
Yes, let's retrain again...

...

📈 Final Accuracy: 1.00
📊 Iterations: 5

Also generates:

accuracy_plot.png showing improvement over rounds

✨ Why This Matters
This small demo reflects core principles of Agentic AI in real-world workflows:

Reasoning

Autonomy

Decision making

Task orchestration

🧠 Built With
Python

Scikit-learn

OpenAI (GPT-3.5)

Matplotlib

dotenv

🔐 Disclaimer
This demo uses an OpenAI key stored locally in .env. Make sure to:

NEVER commit your .env

Use python-dotenv to keep it secure

Made with ❤️ by John Daniel
