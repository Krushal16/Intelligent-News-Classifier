import pandas as pd
from sklearn.model_selection import train_test_split


def load_ag_news_csv(path):
    df = pd.read_csv(path)

    if "Class Index" in df.columns:
        label_col = "Class Index"
    elif "label" in df.columns:
        label_col = "label"
    elif "class_index" in df.columns:
        label_col = "class_index"
    else:
        raise ValueError(f"No label column found. Available columns: {list(df.columns)}")

    if "text" in df.columns:
        text_col = "text"
        df["text"] = df[text_col].fillna("")
    elif "Title" in df.columns and "Description" in df.columns:
        df["text"] = df["Title"].fillna("") + " " + df["Description"].fillna("")
    elif "title" in df.columns and "description" in df.columns:
        df["text"] = df["title"].fillna("") + " " + df["description"].fillna("")
    else:
        raise ValueError(f"No text columns found. Available columns: {list(df.columns)}")

    df = df.rename(columns={label_col: "label"})
    return df[["text", "label"]]


def get_train_test(train_path, test_path, convert_to_zero_based=False):
    train_df = load_ag_news_csv(train_path)
    test_df = load_ag_news_csv(test_path)

    X_train = train_df["text"]
    y_train = train_df["label"]

    X_test = test_df["text"]
    y_test = test_df["label"]

    if convert_to_zero_based:
        y_train = y_train - 1
        y_test = y_test - 1

    return X_train, y_train, X_test, y_test


def get_train_val_test(train_path, test_path, val_size=0.2, random_state=42, convert_to_zero_based=False):
    train_df = load_ag_news_csv(train_path)
    test_df = load_ag_news_csv(test_path)

    X = train_df["text"]
    y = train_df["label"]

    if convert_to_zero_based:
        y = y - 1
        y_test = test_df["label"] - 1
    else:
        y_test = test_df["label"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_size, random_state=random_state, stratify=y
    )

    X_test = test_df["text"]

    return X_train, X_val, X_test, y_train, y_val, y_test