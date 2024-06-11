import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    data = pd.read_csv(file_path)
    X = data.drop(columns=['winner'])
    y = data['winner'].apply(lambda x: 1 if x == 1 else 0)  # Ensure binary encoding
    return train_test_split(X, y, test_size=0.2, random_state=42)
