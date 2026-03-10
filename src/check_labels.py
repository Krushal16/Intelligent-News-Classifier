import pandas as pd
from src.data_loader import get_train_val_test


def main():
    print("Checking raw CSV labels...")

    train_df = pd.read_csv("data/raw/train.csv")

    print("Columns in train.csv:", list(train_df.columns))

    if "label" in train_df.columns:
        label_col = "label"
    elif "Class Index" in train_df.columns:
        label_col = "Class Index"
    elif "class_index" in train_df.columns:
        label_col = "class_index"
    elif "label_id" in train_df.columns:
        label_col = "label_id"
    else:
        raise ValueError(f"No label column found. Available columns: {list(train_df.columns)}")

    print("Using label column:", label_col)
    print("Unique labels in raw train.csv:", sorted(train_df[label_col].unique()))
    print(train_df[label_col].value_counts().sort_index())

    print("\nChecking loader output labels...")
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test(
        "data/raw/train.csv",
        "data/raw/test.csv"
    )

    print("Unique y_train labels:", sorted(pd.Series(y_train).unique()))
    print("Unique y_val labels:", sorted(pd.Series(y_val).unique()))
    print("Unique y_test labels:", sorted(pd.Series(y_test).unique()))

    print("\nLabel check completed.")


if __name__ == "__main__":
    main()