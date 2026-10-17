import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES = ["lag_1","lag_7","lag_14","rollmean_7","rollmean_28","rollstd_7","dow","day_of_year","on_promo"]

def make_features(df):
    parts = []
    for _, group in df.sort_values("date").groupby(["store_id","item_id"], sort=False):
        g = group.sort_values("date").copy()
        g["lag_1"] = g.qty_sold.shift(1)
        g["lag_7"] = g.qty_sold.shift(7)
        g["lag_14"] = g.qty_sold.shift(14)
        past = g.qty_sold.shift(1)
        g["rollmean_7"] = past.rolling(7).mean()
        g["rollmean_28"] = past.rolling(28).mean()
        g["rollstd_7"] = past.rolling(7).std().fillna(0)
        g["dow"] = g.date.dt.dayofweek
        g["day_of_year"] = g.date.dt.dayofyear
        parts.append(g)
    return pd.concat(parts, ignore_index=True).dropna(subset=FEATURES).reset_index(drop=True)


def train_evaluate(df):
    featured = make_features(df)

    # Split by unique dates so the test set contains future dates.
    dates = sorted(featured["date"].dropna().unique())
    if len(dates) < 2:
        raise ValueError("At least two distinct dates are required for evaluation.")

    split_idx = int(len(dates) * 0.8)
    split_idx = min(max(split_idx, 1), len(dates) - 1)
    cut = dates[split_idx]

    train = featured[featured["date"] < cut].copy()
    test = featured[featured["date"] >= cut].copy()

    if train.empty or test.empty:
        raise ValueError("The chronological train/test split produced an empty set.")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(train[FEATURES], train["qty_sold"])

    test["prediction"] = np.maximum(
        0, model.predict(test[FEATURES])
    )
    test["seasonal_naive"] = test["lag_7"].clip(lower=0)

    metrics = {
        "mae": mean_absolute_error(test["qty_sold"], test["prediction"]),
        "rmse": np.sqrt(
            mean_squared_error(test["qty_sold"], test["prediction"])
        ),
        "baseline_mae": mean_absolute_error(
            test["qty_sold"], test["seasonal_naive"]
        ),
        "train_rows": len(train),
        "test_rows": len(test),
        "cut_date": str(pd.Timestamp(cut).date()),
    }

    return model, featured, test, metrics


def recursive_forecast(model, history, horizon=14):
    hist = list(history.sort_values("date").qty_sold.astype(float).values)
    dates = pd.date_range(history.date.max()+pd.Timedelta(days=1), periods=horizon)
    rows = []
    for date in dates:
        vals = np.asarray(hist)
        if len(vals) < 28:
            pred = float(np.mean(vals[-7:]))
        else:
            row = pd.DataFrame([{"lag_1":vals[-1],"lag_7":vals[-7],"lag_14":vals[-14],
                "rollmean_7":vals[-7:].mean(),"rollmean_28":vals[-28:].mean(),
                "rollstd_7":vals[-7:].std(),"dow":date.dayofweek,"day_of_year":date.dayofyear,
                "on_promo":0}])
            pred = float(model.predict(row[FEATURES])[0])
        pred = max(0,pred)
        rows.append({"date":date,"forecast_qty":pred})
        hist.append(pred)
    return pd.DataFrame(rows)
