from pathlib import Path

REQUIRED_FILES = [
    "data/raw/train.csv",
    "data/raw/test.csv",
    "models/best_model.pkl",
    "reports/metrics/week8_best_model_metrics.csv",
    "reports/metrics/week8_misclassified_samples.csv",
    "reports/metrics/week8_confusion_pairs.csv",
    "reports/figures/week8_confusion_matrix_best_model.png",
    "app/app.py",
    "README.md"
]

def main():
    print("=== Week 9 Project Check ===\n")

    missing = []
    for file in REQUIRED_FILES:
        exists = Path(file).exists()
        print(f"{file}: {'FOUND' if exists else 'MISSING'}")
        if not exists:
            missing.append(file)

    print("\nSummary:")
    if not missing:
        print("All required project files are present.")
    else:
        print("Missing files:")
        for f in missing:
            print("-", f)

if __name__ == "__main__":
    main()