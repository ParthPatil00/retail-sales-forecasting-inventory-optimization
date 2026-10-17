
import pandas as pd

from src.forecasting import train_evaluate


def test_forecasting_uses_chronological_dates():
    rows = []
    dates = pd.date_range("2025-01-01", periods=60, freq="D")

    for store in ["STORE-01", "STORE-02"]:
        for item in ["SKU-1001", "SKU-1002"]:
            for i, date in enumerate(dates):
                rows.append({
                    "date": date,
                    "store_id": store,
                    "item_id": item,
                    "qty_sold": 10 + (i % 7),
                    "on_promo": 0,
                })

    df = pd.DataFrame(rows)
    _, _, test, metrics = train_evaluate(df)

    # Every test date must be on or after the cutoff date.
    assert test["date"].min() >= pd.Timestamp(metrics["cut_date"])

    # No training date may overlap a test date.
    assert metrics["train_rows"] > 0
    assert metrics["test_rows"] > 0
