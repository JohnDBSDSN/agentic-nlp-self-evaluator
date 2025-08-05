# generate_better_reviews.py

import pandas as pd
import random
import os

output_dir = "Data"
os.makedirs(output_dir, exist_ok=True)
output_file = os.path.join(output_dir, "input_reviews.csv")

# Positive and negative keywords
positive_keywords = [
    "amazing", "excellent", "love", "great", "happy", "best", "perfect", "fast", "smooth", "wonderful"
]

negative_keywords = [
    "terrible", "horrible", "worst", "bad", "slow", "broken", "not good", "disappointing", "useless", "frustrating"
]

# Neutral/ambiguous phrases to simulate real-world noise
ambiguous_reviews = [
    "It’s okay, not great but not bad.",
    "Decent experience, could be better.",
    "Mixed feelings about this.",
    "Some features are good, others not so much.",
    "I'm not sure how I feel about it."
]

def generate_review(label):
    if label == 1:
        return f"This is a {random.choice(positive_keywords)} product. I am very satisfied."
    else:
        return f"This is a {random.choice(negative_keywords)} product. I am very disappointed."

# Generate 45 positive + 45 negative = 90 clean rows
data = []
for _ in range(45):
    data.append({"text": generate_review(1), "label": 1})
    data.append({"text": generate_review(0), "label": 0})

# Add 10 noisy/ambiguous rows with mixed labels
for _ in range(10):
    review = random.choice(ambiguous_reviews)
    label = random.choice([0, 1])  # Randomly assign class
    data.append({"text": review, "label": label})

# Shuffle and save
random.shuffle(data)
df = pd.DataFrame(data)
df.to_csv(output_file, index=False)

print(f"✅ More realistic dataset saved to: {output_file}")
