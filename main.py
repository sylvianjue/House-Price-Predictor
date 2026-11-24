"""
House Price Predictor - FastAPI app

It uses the model trained in the Colab notebook (house_price_model.ipynb).

Run it:   uvicorn main:app --host 0.0.0.0 --port 8000
Open:     http://127.0.0.1:8000        the web page
          http://127.0.0.1:8000/docs   the automatic API page (try the API there)
"""
import html
import json
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "house_price_model.joblib"
ZIPCODES_PATH = BASE_DIR / "zipcodes.json"
INFO_PATH = BASE_DIR / "model_info.json"

# The data is from 2014-2015, so the model measures the age of a house against 2015
DATA_YEAR = 2015

# Default columns (same order as in the notebook). model_info.json replaces them if it has them.
DEFAULT_NUM = ["bedrooms", "bathrooms", "sqft_living", "sqft_lot", "floors",
               "waterfront", "view", "condition", "grade", "sqft_basement",
               "house_age", "was_renovated", "has_basement"]
DEFAULT_CAT = ["zipcode"]

GRADE_LABELS = {
    3: "3 - Very low", 4: "4 - Low", 5: "5 - Fair", 6: "6 - Below average",
    7: "7 - Average (typical house)", 8: "8 - Good", 9: "9 - Better",
    10: "10 - Very good", 11: "11 - Excellent", 12: "12 - Luxury", 13: "13 - Mansion",
}
VIEW_LABELS = {0: "0 - No view", 1: "1 - Fair", 2: "2 - Average", 3: "3 - Good", 4: "4 - Excellent"}
CONDITION_LABELS = {1: "1 - Poor", 2: "2 - Fair", 3: "3 - Average", 4: "4 - Good", 5: "5 - Very good"}
FLOOR_OPTIONS = [1, 1.5, 2, 2.5, 3, 3.5]


# ---------------------------------------------------------------- load the files
def load_files():
    problems = []
    model, zipcodes, info = None, [], {}

    try:
        info = json.loads(INFO_PATH.read_text())
    except Exception:
        info = {}

    try:
        zipcodes = [str(z) for z in json.loads(ZIPCODES_PATH.read_text())]
    except Exception as error:
        problems.append(f"Could not read zipcodes.json ({error}).")

    try:
        model = joblib.load(MODEL_PATH)
    except Exception as error:
        problems.append(f"Could not load house_price_model.joblib ({error}).")

    return model, zipcodes, info, problems


MODEL, ZIPCODES, INFO, LOAD_PROBLEMS = load_files()
NUM_FEATURES = INFO.get("numeric_features", DEFAULT_NUM)
CAT_FEATURES = INFO.get("categorical_features", DEFAULT_CAT)
TEST_MAE = INFO.get("test_mae")
TEST_R2 = INFO.get("test_r2")


# ---------------------------------------------------------------- the data the API accepts
class HouseInput(BaseModel):
    """The details of one house. FastAPI checks every value before the model sees it."""

    bedrooms: int = Field(3, ge=1, le=10, description="Number of bedrooms")
    bathrooms: float = Field(2.0, ge=0.5, le=8, description="Number of bathrooms (0.5 = a room with a toilet but no shower)")
    sqft_living: int = Field(1800, ge=300, le=13540, description="Living space inside the house (square feet)")
    sqft_lot: int = Field(5000, ge=500, le=1700000, description="Size of the land (square feet)")
    floors: float = Field(1.0, ge=1, le=3.5, description="Number of floors")
    waterfront: int = Field(0, ge=0, le=1, description="1 if the house is on the waterfront, 0 if not")
    view: int = Field(0, ge=0, le=4, description="How good the view is (0 = none, 4 = best)")
    condition: int = Field(3, ge=1, le=5, description="Condition of the house (1 = poor, 5 = very good)")
    grade: int = Field(7, ge=3, le=13, description="Quality of building and design (3 to 13)")
    sqft_basement: int = Field(0, ge=0, le=5000, description="Basement size (square feet), 0 if none")
    yr_built: int = Field(1990, ge=1900, le=DATA_YEAR, description="Year the house was built")
    yr_renovated: int = Field(0, ge=0, le=DATA_YEAR, description="Year it was renovated, 0 if never")
    zipcode: str = Field(..., description="Zipcode (the area). See /zipcodes for the list.")

    model_config = ConfigDict(json_schema_extra={
        "example": {
            "bedrooms": 3, "bathrooms": 2.0, "sqft_living": 1800, "sqft_lot": 5000,
            "floors": 1.0, "waterfront": 0, "view": 0, "condition": 3, "grade": 7,
            "sqft_basement": 0, "yr_built": 1990, "yr_renovated": 0, "zipcode": "98103",
        }
    })

    @field_validator("zipcode", mode="before")
    @classmethod
    def zipcode_as_text(cls, value):
        # a zipcode is an area name, so 98103 and "98103" mean the same thing
        return str(value).strip()


class PredictionOut(BaseModel):
    price: float
    price_text: str
    low: Optional[float] = None
    high: Optional[float] = None
    range_text: Optional[str] = None


def check_house(house):
    """Checks that need more than one value. Returns a list of problems (empty if all is fine)."""
    problems = []
    if ZIPCODES and house.zipcode not in ZIPCODES:
        problems.append("zipcode: please use a zipcode from the list (see /zipcodes).")
    if house.sqft_basement > house.sqft_living:
        problems.append("sqft_basement: the basement cannot be bigger than the living space.")
    if 0 < house.yr_renovated < 1900:
        problems.append("yr_renovated: use 0 (never renovated) or a year from 1900.")
    elif house.yr_renovated and house.yr_renovated < house.yr_built:
        problems.append("yr_renovated: a house cannot be renovated before it was built.")
    return problems


def make_row(house):
    """Makes the one-row table the model expects (same steps as in the notebook)."""
    row = {
        "bedrooms": house.bedrooms,
        "bathrooms": house.bathrooms,
        "sqft_living": house.sqft_living,
        "sqft_lot": house.sqft_lot,
        "floors": house.floors,
        "waterfront": house.waterfront,
        "view": house.view,
        "condition": house.condition,
        "grade": house.grade,
        "sqft_basement": house.sqft_basement,
        "house_age": DATA_YEAR - house.yr_built,
        "was_renovated": 1 if house.yr_renovated > 0 else 0,
        "has_basement": 1 if house.sqft_basement > 0 else 0,
        "zipcode": str(house.zipcode),
    }
    return pd.DataFrame([row], columns=NUM_FEATURES + CAT_FEATURES)


def predict_price(house):
    price = float(MODEL.predict(make_row(house))[0])
    result = {"price": round(price, 2), "price_text": f"${round(price, -3):,.0f}"}
    if TEST_MAE:
        low = max(price - TEST_MAE, 0)
        high = price + TEST_MAE
        result["low"] = round(low, 2)
        result["high"] = round(high, 2)
        result["range_text"] = f"${round(low, -3):,.0f} to ${round(high, -3):,.0f}"
    return result


# ---------------------------------------------------------------- the web page
def options_html(labels, selected):
    return "\n".join(
        f'<option value="{value}"{" selected" if value == selected else ""}>{html.escape(text)}</option>'
        for value, text in labels.items()
    )


def build_page():
    floors = "\n".join(
        f'<option value="{value}"{" selected" if value == 1 else ""}>{value}</option>' for value in FLOOR_OPTIONS
    )
    zipcodes = "\n".join(f'<option value="{html.escape(z)}">{html.escape(z)}</option>' for z in ZIPCODES)

    about = []
    if TEST_R2 is not None:
        about.append(f"<li>On houses it had never seen, it explained about {TEST_R2 * 100:.1f}% of the changes in price (R2).</li>")
    if TEST_MAE:
        about.append(f"<li>On those houses, its guess was off by about ${TEST_MAE:,.0f} on average.</li>")

    banner = ""
    if LOAD_PROBLEMS:
        banner = '<div class="banner"><strong>The model could not be loaded.</strong> ' + html.escape(" ".join(LOAD_PROBLEMS)) + "</div>"

    page = PAGE_TEMPLATE
    page = page.replace("__BANNER__", banner)
    page = page.replace("__FLOORS__", floors)
    page = page.replace("__GRADES__", options_html(GRADE_LABELS, 7))
    page = page.replace("__CONDITIONS__", options_html(CONDITION_LABELS, 3))
    page = page.replace("__VIEWS__", options_html(VIEW_LABELS, 0))
    page = page.replace("__ZIPCODES__", zipcodes)
    page = page.replace("__ABOUT__", "\n".join(about))
    page = page.replace("__DATA_YEAR__", str(DATA_YEAR))
    return page


PAGE_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="description" content="Predict the selling price of a house in King County, USA with a machine learning model.">
<title>House Price Predictor</title>
<style>
  :root {
    --bg: #f3f5fb; --card: #ffffff; --text: #0f172a; --muted: #64748b; --line: #e2e8f0;
    --field: #f8fafc; --brand: #4f46e5; --brand-2: #0ea5e9; --brand-dark: #4338ca;
    --ok: #047857; --ok-bg: #ecfdf5; --bad: #b91c1c; --bad-bg: #fef2f2; --bad-line: #fecaca;
    --shadow: 0 10px 30px rgba(15, 23, 42, 0.08);
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #0b1020; --card: #131a2f; --text: #e8ecf6; --muted: #94a3b8; --line: #243050;
      --field: #0f1629; --brand: #818cf8; --brand-2: #38bdf8; --brand-dark: #6366f1;
      --ok: #6ee7b7; --ok-bg: #0f2a24; --bad: #fca5a5; --bad-bg: #2a1417; --bad-line: #5b2229;
      --shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
    }
  }
  * { box-sizing: border-box; }
  html { -webkit-text-size-adjust: 100%; }
  body { margin: 0; font-family: system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
         background: var(--bg); color: var(--text); line-height: 1.5; }

  /* ---------- top banner ---------- */
  .hero { background: linear-gradient(135deg, #312e81 0%, #4f46e5 45%, #0ea5e9 100%); color: #fff; }
  .hero-inner { max-width: 1040px; margin: 0 auto; padding: 44px 20px 92px; }
  .badges { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 16px; }
  .badge { font-size: 0.78rem; font-weight: 600; padding: 4px 12px; border-radius: 999px;
           background: rgba(255, 255, 255, 0.16); border: 1px solid rgba(255, 255, 255, 0.28); }
  .hero h1 { font-size: clamp(1.9rem, 5vw, 2.8rem); line-height: 1.1; margin: 0 0 10px; letter-spacing: -0.02em; }
  .hero p { margin: 0; max-width: 560px; font-size: 1.05rem; color: rgba(255, 255, 255, 0.88); }

  /* ---------- layout ---------- */
  .wrap { max-width: 1040px; margin: -60px auto 0; padding: 0 20px 48px; }
  .layout { display: grid; grid-template-columns: minmax(0, 1.6fr) minmax(0, 1fr); gap: 20px; align-items: start; }
  @media (max-width: 860px) { .layout { grid-template-columns: 1fr; } .side { position: static; order: -1; } }
  .card { background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 22px; box-shadow: var(--shadow); }
  .side { position: sticky; top: 16px; }

  /* ---------- form ---------- */
  fieldset { border: 0; padding: 0; margin: 0 0 22px; min-width: 0; }
  legend { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 1.02rem; padding: 0; margin-bottom: 12px; }
  legend svg { width: 30px; height: 30px; padding: 6px; border-radius: 10px; color: #fff;
               background: linear-gradient(135deg, var(--brand), var(--brand-2)); }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 14px 16px; }
  @media (max-width: 560px) { .grid { grid-template-columns: 1fr; } }
  .full { grid-column: 1 / -1; }
  label { display: block; font-weight: 600; font-size: 0.88rem; margin-bottom: 5px; }
  .hint { font-weight: 400; color: var(--muted); font-size: 0.78rem; display: block; margin-top: 1px; }
  input, select { width: 100%; padding: 11px 12px; font-size: 1rem; font-family: inherit; border: 1px solid var(--line);
                  border-radius: 12px; background: var(--field); color: var(--text); transition: border-color .15s, box-shadow .15s; }
  input:hover, select:hover { border-color: var(--muted); }
  input:focus, select:focus { outline: none; border-color: var(--brand); box-shadow: 0 0 0 4px rgba(79, 70, 229, 0.18); }
  button { width: 100%; padding: 15px; font-size: 1.05rem; font-weight: 700; font-family: inherit; color: #fff;
           background: linear-gradient(135deg, var(--brand), var(--brand-2)); border: 0; border-radius: 14px; cursor: pointer;
           box-shadow: 0 8px 20px rgba(79, 70, 229, 0.35); transition: transform .12s, box-shadow .12s, opacity .12s; }
  button:hover { transform: translateY(-1px); box-shadow: 0 12px 26px rgba(79, 70, 229, 0.42); }
  button:active { transform: translateY(0); }
  button:focus-visible { outline: 3px solid var(--brand-2); outline-offset: 3px; }
  button:disabled { opacity: 0.65; cursor: wait; transform: none; }

  /* ---------- result ---------- */
  .result-title { font-size: 0.8rem; font-weight: 700; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin: 0 0 8px; }
  .empty { text-align: center; padding: 18px 6px; color: var(--muted); }
  .empty svg { width: 54px; height: 54px; color: var(--brand); opacity: 0.85; margin-bottom: 8px; }
  .empty p { margin: 0; font-size: 0.95rem; }
  .price { font-size: clamp(2.3rem, 6vw, 3rem); font-weight: 800; line-height: 1.1; margin: 0; letter-spacing: -0.02em;
           background: linear-gradient(135deg, var(--brand), var(--brand-2)); -webkit-background-clip: text; background-clip: text;
           -webkit-text-fill-color: transparent; color: var(--brand); }
  .range { margin: 8px 0 14px; color: var(--muted); font-size: 0.92rem; }
  .bar { position: relative; height: 10px; border-radius: 999px; background: linear-gradient(90deg, var(--line), var(--brand-2), var(--line)); margin: 14px 0 6px; }
  .bar i { position: absolute; left: 50%; top: 50%; width: 20px; height: 20px; margin: -10px 0 0 -10px; border-radius: 50%;
           background: #fff; border: 4px solid var(--brand); box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25); }
  .bar-labels { display: flex; justify-content: space-between; font-size: 0.82rem; font-weight: 600; }
  .note { margin: 14px 0 0; font-size: 0.8rem; color: var(--muted); }
  #result-box { animation: pop .35s ease-out; }
  @keyframes pop { from { opacity: 0; transform: translateY(8px) scale(0.98); } to { opacity: 1; transform: none; } }
  @media (prefers-reduced-motion: reduce) { #result-box { animation: none; } button, input, select { transition: none; } }
  .hidden { display: none !important; }
  .banner { background: var(--bad-bg); border: 1px solid var(--bad-line); color: var(--bad); border-radius: 14px;
            padding: 12px 16px; margin-bottom: 16px; font-size: 0.93rem; }
  .banner ul { margin: 4px 0 0; padding-left: 20px; }

  /* ---------- about ---------- */
  .about { margin-top: 20px; }
  .about h2 { font-size: 1.05rem; margin: 0 0 6px; }
  .about p, .about li { color: var(--muted); font-size: 0.9rem; margin: 4px 0; }
  .about ul { padding-left: 20px; margin: 8px 0; }
  .about a { color: var(--brand); }
  footer { text-align: center; color: var(--muted); font-size: 0.8rem; margin-top: 18px; }
</style>
</head>
<body>

<header class="hero">
  <div class="hero-inner">
    <div class="badges">
      <span class="badge">King County, USA</span>
      <span class="badge">Machine Learning</span>
      <span class="badge">Gradient Boosting</span>
    </div>
    <h1>House Price Predictor</h1>
    <p>Fill in the details of a house and the model will estimate its selling price in seconds.</p>
  </div>
</header>

<div class="wrap">
  __BANNER__
  <div id="error-box" class="banner hidden" role="alert"></div>

  <div class="layout">

    <form class="card" id="house-form" novalidate>
      <fieldset>
        <legend>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 11l9-8 9 8"/><path d="M5 10v10h14V10"/><path d="M10 20v-6h4v6"/></svg>
          Size and rooms
        </legend>
        <div class="grid">
          <div>
            <label for="bedrooms">Bedrooms</label>
            <input type="number" id="bedrooms" name="bedrooms" min="1" max="10" step="1" value="3" inputmode="numeric">
          </div>
          <div>
            <label for="bathrooms">Bathrooms <span class="hint">0.5 = toilet but no shower</span></label>
            <input type="number" id="bathrooms" name="bathrooms" min="0.5" max="8" step="0.25" value="2" inputmode="decimal">
          </div>
          <div>
            <label for="sqft_living">Living space (sqft) <span class="hint">Inside the house</span></label>
            <input type="number" id="sqft_living" name="sqft_living" min="300" max="13540" step="1" value="1800" inputmode="numeric">
          </div>
          <div>
            <label for="sqft_lot">Land size (sqft) <span class="hint">The whole plot</span></label>
            <input type="number" id="sqft_lot" name="sqft_lot" min="500" max="1700000" step="1" value="5000" inputmode="numeric">
          </div>
          <div>
            <label for="floors">Floors</label>
            <select id="floors" name="floors">
__FLOORS__
            </select>
          </div>
          <div>
            <label for="sqft_basement">Basement (sqft) <span class="hint">Type 0 if there is none</span></label>
            <input type="number" id="sqft_basement" name="sqft_basement" min="0" max="5000" step="1" value="0" inputmode="numeric">
          </div>
        </div>
      </fieldset>

      <fieldset>
        <legend>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 3l2.7 5.6 6.1.9-4.4 4.3 1 6.1L12 17l-5.4 2.9 1-6.1L3.2 9.5l6.1-.9z"/></svg>
          Quality and surroundings
        </legend>
        <div class="grid">
          <div>
            <label for="grade">Grade <span class="hint">Building and design quality</span></label>
            <select id="grade" name="grade">
__GRADES__
            </select>
          </div>
          <div>
            <label for="condition">Condition</label>
            <select id="condition" name="condition">
__CONDITIONS__
            </select>
          </div>
          <div>
            <label for="view">View</label>
            <select id="view" name="view">
__VIEWS__
            </select>
          </div>
          <div>
            <label for="waterfront">Waterfront</label>
            <select id="waterfront" name="waterfront">
              <option value="0" selected>No</option>
              <option value="1">Yes</option>
            </select>
          </div>
        </div>
      </fieldset>

      <fieldset>
        <legend>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/></svg>
          History
        </legend>
        <div class="grid">
          <div>
            <label for="yr_built">Year built</label>
            <input type="number" id="yr_built" name="yr_built" min="1900" max="__DATA_YEAR__" step="1" value="1990" inputmode="numeric">
          </div>
          <div>
            <label for="yr_renovated">Year renovated <span class="hint">Type 0 if never renovated</span></label>
            <input type="number" id="yr_renovated" name="yr_renovated" min="0" max="__DATA_YEAR__" step="1" value="0" inputmode="numeric">
          </div>
        </div>
      </fieldset>

      <fieldset>
        <legend>
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M12 21s-7-6.2-7-11a7 7 0 0114 0c0 4.8-7 11-7 11z"/><circle cx="12" cy="10" r="2.5"/></svg>
          Location
        </legend>
        <div class="grid">
          <div class="full">
            <label for="zipcode">Zipcode (area)</label>
            <select id="zipcode" name="zipcode">
              <option value="" selected>Choose a zipcode...</option>
__ZIPCODES__
            </select>
          </div>
        </div>
      </fieldset>

      <button type="submit" id="predict-button">Predict price</button>
    </form>

    <aside class="side">
      <div class="card" aria-live="polite">
        <p class="result-title">Estimated selling price</p>

        <div id="result-empty" class="empty">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M3 11l9-8 9 8"/><path d="M5 10v10h14V10"/><path d="M10 20v-6h4v6"/></svg>
          <p>Fill in the form and click <strong>Predict price</strong>. Your estimate will show here.</p>
        </div>

        <div id="result-box" class="hidden">
          <p class="price" id="result-price"></p>
          <p class="range" id="result-range"></p>
          <div id="range-visual" class="hidden">
            <div class="bar"><i></i></div>
            <div class="bar-labels"><span id="range-low"></span><span id="range-high"></span></div>
          </div>
          <p class="note">A guess based on past sales, not an official valuation.</p>
        </div>
      </div>
    </aside>

  </div>

  <div class="card about">
    <h2>About this model</h2>
    <p>A Gradient Boosting regression model trained on more than 21,000 house sales in King County, USA (Kaggle dataset). It runs on a FastAPI server.</p>
    <ul>
__ABOUT__
      <li>The prices come from sales between May 2014 and May 2015, so today's real prices are likely higher.</li>
      <li>It only knows King County. Do not use it for houses in other places.</li>
      <li>It is a learning project and a guess, not an official valuation.</li>
    </ul>
    <p>Developers: the automatic API page is at <a href="docs">/docs</a>.</p>
  </div>

  <footer>Machine Learning project - Python, pandas, scikit-learn, Apache Airflow, Hyperopt, FastAPI</footer>
</div>

<script>
  var form = document.getElementById("house-form");
  var errorBox = document.getElementById("error-box");
  var resultBox = document.getElementById("result-box");
  var resultEmpty = document.getElementById("result-empty");
  var rangeVisual = document.getElementById("range-visual");
  var button = document.getElementById("predict-button");

  var LABELS = {
    bedrooms: "Bedrooms", bathrooms: "Bathrooms", sqft_living: "Living space", sqft_lot: "Land size",
    floors: "Floors", waterfront: "Waterfront", view: "View", condition: "Condition", grade: "Grade",
    sqft_basement: "Basement size", yr_built: "Year built", yr_renovated: "Year renovated", zipcode: "Zipcode"
  };

  function hideResult() {
    resultBox.classList.add("hidden");
    resultEmpty.classList.remove("hidden");
  }

  function showErrors(messages) {
    hideResult();
    errorBox.textContent = "";
    var list = document.createElement("ul");
    messages.forEach(function (message) {
      var item = document.createElement("li");
      item.textContent = message;
      list.appendChild(item);
    });
    errorBox.appendChild(list);
    errorBox.classList.remove("hidden");
    errorBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // The server sends errors either as a text, or as a list (of texts or of objects with loc and msg)
  function readErrors(data) {
    var detail = data && data.detail;
    if (typeof detail === "string") { return [detail]; }
    if (Array.isArray(detail)) {
      return detail.map(function (item) {
        if (typeof item === "string") { return item; }
        var where = item.loc && item.loc.length ? item.loc[item.loc.length - 1] : "";
        var name = LABELS[where] || where;
        return (name ? name + ": " : "") + (item.msg || "not valid");
      });
    }
    return ["Something went wrong. Please check the values and try again."];
  }

  function showResult(data) {
    document.getElementById("result-price").textContent = data.price_text;
    document.getElementById("result-range").textContent = data.range_text ? "Likely range: " + data.range_text : "";
    if (data.range_text) {
      var parts = data.range_text.split(" to ");
      document.getElementById("range-low").textContent = parts[0] || "";
      document.getElementById("range-high").textContent = parts[1] || "";
      rangeVisual.classList.remove("hidden");
    } else {
      rangeVisual.classList.add("hidden");
    }
    resultEmpty.classList.add("hidden");
    resultBox.classList.remove("hidden");
    // restart the small pop animation
    resultBox.style.animation = "none";
    void resultBox.offsetWidth;
    resultBox.style.animation = "";
    if (window.matchMedia("(max-width: 860px)").matches) {
      resultBox.scrollIntoView({ behavior: "smooth", block: "center" });
    }
  }

  form.addEventListener("submit", function (event) {
    event.preventDefault();
    errorBox.classList.add("hidden");

    var payload = {};
    var missing = [];
    Object.keys(LABELS).forEach(function (name) {
      var raw = String(form.elements[name].value).trim().replace(/,/g, "");
      if (raw === "" && name === "yr_renovated") { raw = "0"; }
      if (raw === "") { missing.push(LABELS[name] + ": please fill this in."); return; }
      payload[name] = (name === "zipcode") ? raw : Number(raw);
    });
    if (missing.length) { showErrors(missing); return; }

    button.disabled = true;
    button.textContent = "Predicting...";

    fetch("predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    })
      .then(function (response) {
        return response.json().then(function (data) { return { ok: response.ok, data: data }; });
      })
      .then(function (answer) {
        if (!answer.ok) { showErrors(readErrors(answer.data)); return; }
        showResult(answer.data);
      })
      .catch(function () {
        showErrors(["Could not reach the server. Please try again."]);
      })
      .then(function () {
        button.disabled = false;
        button.textContent = "Predict price";
      });
  });
</script>
</body>
</html>
"""

PAGE_HTML = build_page()


# ---------------------------------------------------------------- the API
app = FastAPI(
    title="House Price Predictor",
    description="Predicts the selling price of a house in King County, USA, from simple details about the house.",
    version="1.0.0",
)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def home():
    return HTMLResponse(PAGE_HTML)


@app.get("/health")
def health():
    """Tells if the server is running and the model is loaded."""
    loaded = MODEL is not None and bool(ZIPCODES)
    return {"status": "ok" if loaded else "model not loaded"}


@app.get("/zipcodes")
def zipcodes():
    """The zipcodes the model knows."""
    return {"zipcodes": ZIPCODES}


@app.get("/model-info")
def model_info():
    """The scores and the settings of the model."""
    return INFO


@app.post("/predict", response_model=PredictionOut)
def predict(house: HouseInput):
    """Send the details of a house and get its estimated selling price."""
    if MODEL is None or not ZIPCODES:
        raise HTTPException(status_code=503, detail="The model is not loaded on the server.")

    problems = check_house(house)
    if problems:
        raise HTTPException(status_code=422, detail=problems)

    try:
        return predict_price(house)
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {error}")