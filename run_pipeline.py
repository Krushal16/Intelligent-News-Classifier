# run_pipeline.py
from src.data_loader import get_train_val_test
from src.preprocess import apply_cleaning
from src.train import train_models
from src.evaluate import evaluate_model


def main():
    # 1. Load data
    X_train, X_val, X_test, y_train, y_val, y_test = get_train_val_test(
        "data/raw/train.csv",
        "data/raw/test.csv"
    )

    # 2. Clean text
    X_train_clean = apply_cleaning(X_train)
    X_val_clean = apply_cleaning(X_val)
    X_test_clean = apply_cleaning(X_test)

    # For now we train on full training set (train only),
    # and keep val for future tuning.
    # 3. Train models
    model_paths = train_models(X_train_clean, y_train)

    # 4. Evaluate each model on test set
    for name, path in model_paths.items():
        evaluate_model(
            model_path=path,
            X_test=X_test_clean,
            y_test=y_test,
            model_name=name
        )


if __name__ == "__main__":
    main()
