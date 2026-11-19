# Adaptive CWI Imputer

Web app that fills missing numeric values in any uploaded CSV/Excel file.
Numeric columns are detected automatically; for each column the app tests several imputers
(mean, median, KNN, regression, CWI-kNN) on the user's own data and uses the best one.

## Run
```
python -m venv .venv
.venv\Scripts\activate          # Windows   (Mac/Linux: source .venv/bin/activate)
pip install -r requirements.txt
streamlit run app.py
```

## Structure
```
app.py              Streamlit interface
imputer/cwi_knn.py  The method (column detection, CWI-kNN v2, selection)
requirements.txt
```
