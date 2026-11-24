# House Price Predictor

A machine learning project that predicts the selling price of a house in King County, USA (the Seattle area).
The model is trained on the Kaggle dataset **House Sales in King County, USA** (more than 21,000 sales from May 2014 to May 2015) and deployed as a **FastAPI** web app.

**Live app:** https://YOUR-RENDER-LINK.onrender.com
(The free plan sleeps when nobody uses it, so the first load can take about a minute.)

## Results (on data the model never saw)

| Metric | Score |
|---|---|
| R2 | 0.881 |
| MAE | about $70,369 |
| RMSE | about $116,658 |

## How it was built

1. **Data pipeline:** Apache Airflow DAG (`load_data`, `check_data`, `clean_data`, `make_features`, `split_data`)
2. **Data preparation:** cleaning, outliers (IQR rule), imputation, bias check, new features (house age, renovated, basement)
3. **Model:** Gradient Boosted Decision Trees (a tree-based ensemble regression model), trained on the log of the price
4. **Tuning:** RandomizedSearchCV, then Hyperopt to find the parameters, then GridSearchCV to improve the accuracy
5. **Explainable AI:** SHAP to explain why the model gives each price
6. **Deployment:** FastAPI on Render

## Files in this repository

| File | What it does |
|---|---|
| `main.py` | The FastAPI app (the web page and the API) |
| `house_price_model.joblib` | The trained model (with all preprocessing inside) |
| `zipcodes.json` | The zipcodes for the dropdown on the website |
| `model_info.json` | The scores, features and best settings of the model |
| `requirements.txt` | The libraries the app needs |
| `.python-version` | The Python version used on Render (3.12.8) |

## Pages and API

| Address | What it does |
|---|---|
| `/` | The website with the form |
| `/predict` | `POST` the house details as JSON and get the price back |
| `/health` | Checks that the app and the model are running |
| `/zipcodes` | The list of zipcodes the model knows |
| `/model-info` | The model scores and settings |
| `/docs` | The automatic API page, where you can try it |

Example request:

```
POST /predict
Content-Type: application/json

{"bedrooms": 3, "bathrooms": 2, "sqft_living": 1800, "sqft_lot": 5000, "floors": 1,
 "waterfront": 0, "view": 0, "condition": 3, "grade": 7, "sqft_basement": 0,
 "yr_built": 1990, "yr_renovated": 0, "zipcode": "98103"}
```

It returns the price, a likely range (the price plus or minus the MAE) and the same range as text.

## Run it on your computer

```
pip install -r requirements.txt
uvicorn main:app --reload
```

Then open http://127.0.0.1:8000

## Deploy on Render

- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn main:app --host 0.0.0.0 --port $PORT`
- **Health Check Path:** `/health`

## Limits

- The prices are from 2014 to 2015, so today's real prices are likely higher.
- The model only knows King County. Do not use it for other places.
- It is a learning project and the price is an estimate, not an official valuation.
