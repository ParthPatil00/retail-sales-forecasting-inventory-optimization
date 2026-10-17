import numpy as np
import pandas as pd

def generate_retail_data(days=420, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.date_range(end=pd.Timestamp.today().normalize()-pd.Timedelta(days=1), periods=days)
    products = [
        ("SKU-1001","Coffee","Beverages",95,4), ("SKU-1002","Rice 5kg","Grocery",260,5),
        ("SKU-1003","Shampoo","Personal Care",125,6), ("SKU-1004","LED Bulb","Electronics",85,8),
        ("SKU-1005","Storage Box","Home",180,7), ("SKU-1006","Tea","Beverages",70,4),
        ("SKU-1007","Biscuits","Grocery",25,3), ("SKU-1008","Detergent","Home",145,6)]
    rows = []
    for si, sf in enumerate([0.85, 1.0, 1.18], 1):
        for sku, name, cat, price, lead in products:
            base = rng.uniform(5, 25)*sf
            for i, date in enumerate(dates):
                promo = int(rng.random() < .09)
                weekend = 1.18 if date.dayofweek >= 5 else 1
                seasonal = 1 + .12*np.sin(2*np.pi*i/365.25)
                expected = base*weekend*seasonal*(1+.00035*i)*(1.35 if promo else 1)
                qty = int(rng.poisson(max(expected, .1)))
                stockout = int(rng.random() < .018)
                if stockout: qty = int(qty*.15)
                rows.append({"date":date, "store_id":f"STORE-{si:02d}", "item_id":sku,
                    "product_name":name, "category":cat, "qty_sold":qty, "price":price,
                    "on_promo":promo, "stockout_flag":stockout, "unit_cost":round(price*.62,2),
                    "holding_cost_rate":.20, "ordering_cost":350.0, "lead_time_days":lead,
                    "stock_on_hand":int(rng.integers(0,180))})
    return pd.DataFrame(rows)

def quality_report(df):
    return {"rows":len(df), "duplicate_store_item_dates":int(df.duplicated(["store_id","item_id","date"]).sum()),
            "missing_qty_sold":int(df.qty_sold.isna().sum()), "negative_qty_sold":int((df.qty_sold<0).sum()),
            "stockout_flagged_rows":int(df.stockout_flag.sum())}
