import pandas as pd
import os

os.makedirs("reports/metrics", exist_ok=True)

def main():
    tech_rows = [
        {"technology": "Python", "purpose": "Programming language for project implementation"},
        {"technology": "Pandas", "purpose": "Data loading and manipulation"},
        {"technology": "Scikit-learn", "purpose": "TF-IDF vectorization, model training, evaluation"},
        {"technology": "Joblib", "purpose": "Model saving and loading"},
        {"technology": "Matplotlib", "purpose": "Confusion matrix plotting"},
        {"technology": "Seaborn", "purpose": "Visualization for evaluation outputs"},
        {"technology": "Streamlit", "purpose": "Web demo for news classification"},
        {"technology": "Jupyter Notebook", "purpose": "Experiment tracking and notebook-based evaluation"},
        {"technology": "GitHub", "purpose": "Version control and collaboration"}
    ]

    df = pd.DataFrame(tech_rows)
    df.to_csv("reports/metrics/week9_technology_table.csv", index=False)

    print(df.to_string(index=False))
    print("\nSaved: reports/metrics/week9_technology_table.csv")

if __name__ == "__main__":
    main()