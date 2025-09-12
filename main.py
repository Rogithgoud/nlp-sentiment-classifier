import pandas as pd
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("data/Reviews.csv")

# Keep only non-neutral reviews
df = df[df['Score'] != 3]

# Map scores to binary labels
df['label'] = df['Score'].apply(lambda x: 1 if x > 3 else 0)

# Sample smaller dataset
df_small = df.sample(2000, random_state=42)

# Train-test split
train_x, test_x, train_y, test_y = train_test_split(
    df_small['Text'], df_small['label'], test_size=0.2, random_state=42
)

print("Training samples:", len(train_x))
print("Testing samples:", len(test_x))
