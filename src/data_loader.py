# src/data_loader.py
import os
import pandas as pd
from typing import Tuple
from sklearn.model_selection import train_test_split

LABEL_MAP = {
    1: "World",
    2: "Sports",
    3: "Business",
    4: "Sci/Tech"
}

def load_ag_news_csv(
    csv_path: str = "data/raw/ag_news.csv"
) -> pd.DataFrame:
    """
    Load AG News CSV downloaded from Kaggle.

    Expected columns (adjust if different):
    - class_index (int)
    - title (str)
    - description (str)
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found at {csv_path}. Put Kaggle file there.")

    df = pd.read_csv(csv_path)
    # Normalize column names to lowercase and remove spaces
    df.columns = df.columns.str.lower().str.replace(' ', '_')
    # Inspect columns once manually to confirm
    # print(df.head())

    # Combine title + description into one text field
    if "title" in df.columns and "description" in df.columns:
        df["text"] = df["title"].astype(str) + " " + df["description"].astype(str)
    elif "text" not in df.columns:
        raise ValueError("CSV must have 'title' and 'description' or a 'text' column.")

    # Map numeric labels to names
    if "class_index" in df.columns:
        df["label_id"] = df["class_index"]
        df["label"] = df["class_index"].map(LABEL_MAP)
    elif "class" in df.columns:
        # Handle "Class" column (capitalized)
        df["label_id"] = df["class"]
        df["label"] = df["class"].map(LABEL_MAP)
    elif "label" in df.columns:
        # if already string labels, just copy
        df["label_id"] = df["label"].astype("category").cat.codes
    else:
        raise ValueError("CSV must have 'class_index' or 'label' column.")

    return df

def get_train_val_test(
    train_csv: str = "data/raw/train.csv",
    test_csv: str = "data/raw/test.csv",
    val_size: float = 0.2,
    random_state: int = 42
):
    train_df = load_ag_news_csv(train_csv)
    test_df = load_ag_news_csv(test_csv)

    X = train_df["text"]
    y = train_df["label_id"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=val_size,
        random_state=random_state,
        stratify=y
    )

    X_test = test_df["text"]
    y_test = test_df["label_id"]

    return X_train, X_val, X_test, y_train, y_val, y_test

# src/data_loader.py
def quick_split_stats(y_train, y_val, y_test):
    import pandas as pd
    return pd.DataFrame({
        "train": pd.Series(y_train).value_counts().sort_index(),
        "val": pd.Series(y_val).value_counts().sort_index(),
        "test": pd.Series(y_test).value_counts().sort_index()
    }).fillna(0).astype(int)