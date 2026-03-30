from pathlib import Path

REQUIRED_FILES = [
    "README.md",
    "requirements.txt",
    "run_pipeline.py",
    "app/app.py",
    "src/data_loader.py",
    "src/preprocess.py",
    "models/best_model.pkl",
    "reports/metrics/week8_best_model_metrics.csv",
    "reports/metrics/week8_confusion_pairs.csv",
    "reports/metrics/week9_results_summary.csv",
    "reports/figures/week8_confusion_matrix_best_model.png",
    "data/raw/train.csv",
    "data/raw/test.csv"
]

def main():
    print("=== Week 10 Final Submission Check ===\n")

    missing = []
    for file_path in REQUIRED_FILES:
        exists = Path(file_path).exists()
        print(f"{file_path}: {'FOUND' if exists else 'MISSING'}")
        if not exists:
            missing.append(file_path)

    print("\nSummary")
    if missing:
        print("Missing files:")
        for item in missing:
            print("-", item)
    else:
        print("All required files are present. Project is submission-ready.")

if __name__ == "__main__":
    main()