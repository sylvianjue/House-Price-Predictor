# House Price Predictor (Flask)

A small website that predicts the selling price of a house in King County, USA.
It uses the model trained in the Colab notebook `house_price_model.ipynb`.

## What is in this folder

| File | What it does |
|---|---|
| `app.py` | The Flask website (checks the inputs, loads the model, predicts) |
| `templates/index.html` | The web page with the form |
| `requirements.txt` | Libraries the website needs (versions match the Colab notebook) |
| `.python-version` | Tells Render to use Python 3.12.8 |
| `Procfile` | Tells the host how to start the website (`gunicorn app:app`) |
| `render.yaml` | Optional settings for Render |

**3 files come from Colab and are NOT in this folder yet:**
`house_price_model.joblib`, `zipcodes.json`, `model_info.json`

> Do NOT copy the `requirements.txt` that Colab gives you. Keep the one in this folder.
> It has the same scikit-learn version as Colab (1.6.1) and matching pandas and numpy versions, which avoids version errors.

---

## Part A. Get the model files from Colab

1. In your Colab notebook, run all cells. The last cells make and download **deploy_files.zip**.
2. Unzip it on your computer. You get 4 files.
3. Copy these 3 into this folder, next to `app.py`:
   - `house_price_model.joblib`
   - `zipcodes.json`
   - `model_info.json`
4. Ignore the 4th file (`requirements.txt`).

The folder must now look like this:

```
app/
  app.py
  Procfile
  render.yaml
  requirements.txt
  .python-version
  house_price_model.joblib   <- from Colab
  zipcodes.json              <- from Colab
  model_info.json            <- from Colab
  templates/
    index.html
```

## Part B. (Optional) Test it on your computer

```
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000, choose a zipcode and click **Predict price**.

## Part C. Put the code on GitHub

1. Make a free account at github.com.
2. Click **New repository**. Name it `house-price-predictor`. Choose **Public**. Click **Create repository**.
3. Click **uploading an existing file**.
4. Open the `app` folder on your computer, select everything inside it, and drag it into the GitHub page. Include the `templates` folder and the hidden file `.python-version` (if you cannot see it, turn on "show hidden files", or create it on GitHub with **Add file > Create new file**, name `.python-version`, content `3.12.8`).
5. Click **Commit changes**.

Check that `app.py` and `house_price_model.joblib` are at the top level of the repository, not inside another folder.

## Part D. Deploy for free on Render

1. Make a free account at render.com (sign in with GitHub is easiest).
2. Click **New +** then **Web Service**.
3. Connect GitHub and choose the `house-price-predictor` repository.
4. Fill in the settings:
   - **Language:** Python 3
   - **Branch:** main
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Instance Type:** Free
5. Click **Create Web Service**. Wait a few minutes until the log says the service is live.
6. Click the link at the top (like `https://house-price-predictor-xxxx.onrender.com`). That is your public website.
7. Open `your-link/health`. It should show `{"status":"ok"}`.

The free plan puts the site to sleep when nobody uses it. The first visit after sleeping can take about a minute, so open your link a few minutes before you present.

## If something goes wrong

| Problem | Fix |
|---|---|
| Page says "The model could not be loaded" | `house_price_model.joblib` or `zipcodes.json` is missing from the top level of the repository. Upload them. |
| Build fails on `numpy` or `scikit-learn` | Render is using the wrong Python. In Render open **Environment**, add `PYTHON_VERSION` = `3.12.8`, then **Manual Deploy > Deploy latest commit**. |
| Model loads but gives an error when predicting | The model and the library versions do not match. Check that `requirements.txt` still says `scikit-learn==1.6.1`, and that the model file came from the same Colab run. |
| You re-ran Colab and the numbers changed | Upload all 3 new model files together (replace the old ones on GitHub). Render rebuilds by itself. |
| Colab shows a different scikit-learn version next time | Change the number in `requirements.txt` to the one Colab prints in its first cell. |

## JSON address (optional)

```
POST /api/predict
Content-Type: application/json

{"bedrooms":3,"bathrooms":2,"sqft_living":1800,"sqft_lot":5000,"floors":1,
 "waterfront":0,"view":0,"condition":3,"grade":7,"sqft_basement":0,
 "yr_built":1990,"yr_renovated":0,"zipcode":"98103"}
```

It returns `{"price": ..., "price_text": "$...", "range_text": "$... to $..."}`.
Use any zipcode that is in `zipcodes.json`.
