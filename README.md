# Intelligent-News-Classifier
Group 1 - AI/ML project

For clonning project
git clone github_project_link

git branch - to get list of branches
git status - to check status of local project
git pull - to get latest code

git config --global user.email "you@example.com"
git config --global user.name "Your Name"

git checkout dev
python -m venv venv

# Windows PowerShell (recommended)
```powershell
# If scripts are blocked, enable for current user (no admin required):
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Activate the virtual environment (dot-source to affect current session):
 .\venv\Scripts\activate
```

# to install new library
pip install -r requirements.txt

# to run pipeline
python run_pipeline.py

# to run project
streamlit run app/app.py

# week 4 model comparision 
python -m src.tune
# or
python src/tune.py

## Label Encoding
This project uses the AG News dataset label mapping consistently across all modules.

- 1 = World
- 2 = Sports
- 3 = Business
- 4 = Sci/Tech

The same encoding is used in:
- data loading
- model training
- evaluation
- misclassified sample reports
- Streamlit demo