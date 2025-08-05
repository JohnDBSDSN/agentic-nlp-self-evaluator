import pandas as pd
from sklearn.metrics import accuracy_score
from openai import OpenAI


def load_mock_data():
    """
    Loads review data from a CSV file: Data/input_reviews.csv
    Expected columns: text, label
    """
    file_path = "Data/input_reviews.csv"
    df = pd.read_csv(file_path)

    # Basic validation
    assert 'text' in df.columns and 'label' in df.columns, "CSV must contain 'text' and 'label' columns"
    return df


def get_accuracy(y_true, y_pred):
    """
    Computes classification accuracy.
    """
    return accuracy_score(y_true, y_pred)


def query_agent(accuracy: float, threshold: float = 0.95) -> str:
    """
    Asks the GPT agent whether to retrain the model based on current accuracy.
    Returns the agent's natural language decision and explanation.
    """
    client = OpenAI()  # Automatically picks up OPENAI_API_KEY from environment

    prompt = (
        f"The current model accuracy is {accuracy:.2f}.\n"
        f"The threshold for acceptable performance is {threshold:.2f}.\n"
        "Should we retrain the model again to improve accuracy, or stop now?\n"
        "Please respond with 'yes' or 'no' and provide a reason."
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message.content
