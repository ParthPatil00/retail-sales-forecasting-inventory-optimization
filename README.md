# Retail Sales Forecasting & Inventory Optimization System

A professional portfolio starter combining simulated retail data, demand forecasting, inventory policy calculations, and an interactive Streamlit dashboard.

> **Data note:** All sales data is synthetic. Prices, costs, lead times, and inventory values are simulated assumptions, not company data or verified savings.

## Features
- Synthetic multi-store, multi-SKU daily demand with trend, weekly patterns, promotions, and stockout flags
- Data-quality checks and sales EDA
- Lagged and rolling-window features
- Random Forest regression with chronological holdout and seasonal-naive baseline
- Safety stock, reorder point (ROP), EOQ, and suggested order calculations
- Interactive dashboard, filters, forecast chart, and CSV export

## Tech stack
Python, Pandas, NumPy, Matplotlib, scikit-learn, SciPy, Streamlit.

## Architecture
```text
Synthetic data → Quality checks → EDA/features → Chronological evaluation
                                                   ↓
Dashboard ← CSV action queue ← Inventory policy ← Forecasts + residuals
```

## Folder structure
```text
app.py
src/data.py
src/forecasting.py
src/inventory.py
tests/test_inventory.py
requirements.txt
README.md
.gitignore
images/
outputs/
```

## Install and run

Python 3.11 or 3.12 is recommended.

### Windows PowerShell
```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```
If activation is blocked, use Command Prompt: `.venv\Scripts\activate.bat`

### macOS / Linux
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

## Dashboard workflow
1. Adjust historical days, simulation seed, target service level, and forecast horizon.
2. Filter by store and category.
3. Inspect sales KPIs and demand patterns.
4. Review chronological holdout metrics versus a seasonal-naive baseline.
5. Select a SKU to view its forecast and replenishment policy.
6. Export the action queue to CSV.

## Methodology
Features include lags (1, 7, 14 days), shifted rolling statistics, weekday, day-of-year, and promotion flag. Evaluation uses a chronological holdout rather than random splitting. Inventory estimates use lead-time demand, service-level z-score safety stock, ROP, and classic EOQ.

## Limitations
This is an educational starter. Stockout rows are flagged but not corrected for censored demand. The residual-based uncertainty estimate is simplified. Production use requires rolling-origin backtesting, supplier and pack-size constraints, open purchase orders, shelf-life rules, validated demand, and planner review.

## Screenshots
Save dashboard screenshots to `images/` and embed them here, e.g. `![Dashboard](images/dashboard_overview.png)`.

## Future improvements
Intermittent-demand Croston/SBA model, rolling-origin backtests and MASE, promotion/price modeling, expiry-aware inventory, FastAPI endpoint, model monitoring, and deployment.

## Author
Replace this section with your name and GitHub profile before publishing.
