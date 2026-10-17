from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT))
from src.data import generate_retail_data, quality_report
from src.forecasting import train_evaluate, recursive_forecast
from src.inventory import inventory_policy

st.set_page_config(page_title="RetailIQ | Forecast & Inventory", page_icon="📦", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif}
.stApp{background:#F5F7FB}.block-container{padding-top:1.5rem;max-width:1500px}
[data-testid="stSidebar"]{background:#111827}[data-testid="stSidebar"] *{color:#E5E7EB}
.hero{background:linear-gradient(120deg,#101827,#173B55 65%,#0F766E);border-radius:18px;padding:27px 30px;color:white;margin-bottom:20px}
.hero h1{font-family:Manrope,sans-serif;font-size:2rem;font-weight:800;margin:0;color:white}
.hero p{color:#D1E5EB;margin:7px 0 0;font-size:1rem}
.eyebrow{text-transform:uppercase;letter-spacing:2px;font-size:.72rem;color:#7DD3C7;font-weight:700;margin-bottom:9px}
.kpi{background:white;border:1px solid #E5EAF2;border-radius:14px;padding:17px 18px;min-height:108px}
.kpi-label{color:#64748B;font-size:.78rem;font-weight:700}.kpi-value{color:#111827;font-family:Manrope,sans-serif;font-size:1.55rem;font-weight:800;margin-top:8px}
.kpi-note{color:#0F766E;font-size:.75rem;margin-top:3px}
.section-head{color:#111827;font-family:Manrope,sans-serif;font-weight:800;font-size:1.18rem;margin:14px 0 3px}
.section-sub{color:#64748B;font-size:.86rem;margin-bottom:12px}

/* Streamlit metric cards: improve contrast */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #D9E2EF;
    border-radius: 12px;
    padding: 12px 15px;
}

/* Metric labels */
div[data-testid="stMetric"] label,
div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p {
    color: #475569 !important;
    opacity: 1 !important;
}

/* Metric values */
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] > div,
div[data-testid="stMetricValue"] p {
    color: #111827 !important;
    opacity: 1 !important;
    font-weight: 800 !important;
}

/* Metric change indicators and supporting text */
div[data-testid="stMetricDelta"],
div[data-testid="stMetricDelta"] svg {
    opacity: 1 !important;
}

</style>
""",unsafe_allow_html=True)

@st.cache_data
def load_data(days,seed): return generate_retail_data(days,seed)
@st.cache_data(show_spinner="Training forecasting model...")
def evaluate(df): return train_evaluate(df)

with st.sidebar:
    st.markdown("## 📦 RetailIQ")
    st.caption("DEMAND • INVENTORY • DECISIONS")
    st.divider()
    st.markdown("### Simulation controls")
    days=st.slider("Historical days",180,540,420,30)
    seed=st.number_input("Simulation seed",1,9999,42)
    service=st.select_slider("Target service level",[.90,.95,.98,.99],value=.95,format_func=lambda x:f"{x:.0%}")
    horizon=st.slider("Forecast horizon (days)",7,30,14)
    st.divider()
    st.caption("Portfolio simulation · Synthetic data")

data=load_data(days,int(seed))
quality=quality_report(data)
model,featured,test,metrics=evaluate(data)
st.markdown('<div class="hero"><div class="eyebrow">RETAIL OPERATIONS INTELLIGENCE</div><h1>Forecast demand. Optimize inventory.</h1><p>A decision-support workspace for sales performance, demand planning, and replenishment.</p></div>',unsafe_allow_html=True)

a,b,c=st.columns([1,1,1.4])
with a: store=st.selectbox("Store",["All stores"]+sorted(data.store_id.unique()))
with b: category=st.selectbox("Category",["All categories"]+sorted(data.category.unique()))
with c: st.markdown(f"<div style='padding-top:30px;color:#64748B'>● Simulation active &nbsp; · &nbsp; Last historical date: {data.date.max().date()}</div>",unsafe_allow_html=True)
df=data.copy()
if store!="All stores": df=df[df.store_id==store]
if category!="All categories": df=df[df.category==category]
kpis=st.columns(4)
vals=[("UNITS SOLD",f"{df.qty_sold.sum():,}","Across selected scope"),
("SIMULATED SALES VALUE",f"₹{(df.qty_sold*df.price).sum()/100000:.2f}L","Units × assumed price"),
("ACTIVE PRODUCTS",str(df.item_id.nunique()),"Unique items in scope"),
("STOCKOUT-FLAG RATE",f"{df.stockout_flag.mean()*100:.1f}%","Synthetic flags, not lost-sales estimate")]
for col,(label,value,note) in zip(kpis,vals):
    col.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',unsafe_allow_html=True)

st.markdown('<div class="section-head">Sales performance</div><div class="section-sub">Historical demand patterns and category contribution</div>',unsafe_allow_html=True)
left,right=st.columns([1.6,1])
with left:
    daily=df.groupby("date",as_index=False).qty_sold.sum()
    fig,ax=plt.subplots(figsize=(8,3.2)); ax.plot(daily.date,daily.qty_sold,linewidth=1.7)
    ax.fill_between(daily.date,daily.qty_sold,alpha=.10); ax.set_ylabel("Units")
    ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",alpha=.2); fig.tight_layout()
    st.pyplot(fig,use_container_width=True); plt.close(fig)
with right:
    cat=df.groupby("category").qty_sold.sum().sort_values()
    fig,ax=plt.subplots(figsize=(5,3.2)); ax.barh(cat.index,cat.values); ax.set_xlabel("Units sold")
    ax.spines[["top","right","left"]].set_visible(False); ax.grid(axis="x",alpha=.2); fig.tight_layout()
    st.pyplot(fig,use_container_width=True); plt.close(fig)

st.markdown('<div class="section-head">Forecast quality</div><div class="section-sub">Chronological holdout versus last-week seasonal-naive baseline</div>',unsafe_allow_html=True)
m=st.columns(4)
m[0].metric("Model MAE",f"{metrics['mae']:.2f} units")
m[1].metric("Model RMSE",f"{metrics['rmse']:.2f} units")
m[2].metric("Baseline MAE",f"{metrics['baseline_mae']:.2f} units")
imp=(1-metrics["mae"]/metrics["baseline_mae"])*100 if metrics["baseline_mae"] else 0
m[3].metric("MAE vs baseline",f"{imp:+.1f}%")
st.caption(f"Train rows: {metrics['train_rows']:,} · Test rows: {metrics['test_rows']:,} · Split date: {metrics['cut_date']}. Aggregate metrics can hide SKU-level variation.")

default_store=store if store!="All stores" else "STORE-02"
choices=data[data.store_id==default_store]
if category!="All categories": choices=choices[choices.category==category]
if choices.empty:
    choices=data[data.store_id==default_store]
sku=st.selectbox("Product for forecast and replenishment",sorted(choices.item_id.unique()))
hist=data[(data.store_id==default_store)&(data.item_id==sku)].sort_values("date")
forecast=recursive_forecast(model,hist,horizon)
resid_std=float((test.qty_sold-test.prediction).std())
last=hist.iloc[-1]
pol=inventory_policy(forecast.forecast_qty.values,resid_std,int(last.stock_on_hand),int(last.lead_time_days),
    float(service),float(hist.qty_sold.mean()*365),float(last.ordering_cost),float(last.unit_cost),float(last.holding_cost_rate))

l,r=st.columns([1.3,1])
with l:
    st.markdown('<div class="section-head">Demand outlook</div><div class="section-sub">Recent history and iterative daily forecast</div>',unsafe_allow_html=True)
    recent=hist.tail(60)
    fig,ax=plt.subplots(figsize=(8,3.5)); ax.plot(recent.date,recent.qty_sold,label="Historical actual")
    ax.plot(forecast.date,forecast.forecast_qty,label="Forecast",linestyle="--",linewidth=2)
    ax.axvline(hist.date.max(),linestyle=":",alpha=.7); ax.set_ylabel("Units"); ax.legend(frameon=False)
    ax.spines[["top","right"]].set_visible(False); ax.grid(axis="y",alpha=.2); fig.tight_layout()
    st.pyplot(fig,use_container_width=True); plt.close(fig)
    st.dataframe(forecast.assign(forecast_qty=forecast.forecast_qty.round(1)),hide_index=True,use_container_width=True)
with r:
    st.markdown('<div class="section-head">Replenishment recommendation</div><div class="section-sub">Policy estimates using simulated costs and uncertainty</div>',unsafe_allow_html=True)
    status="REORDER REVIEW" if pol["reorder_now"] else "STOCK ABOVE ROP"
    color="#B45309" if pol["reorder_now"] else "#0F766E"
    st.markdown(f'<div style="background:white;border:1px solid #E5EAF2;border-radius:14px;padding:18px"><div style="font-size:.76rem;color:#64748B;font-weight:700">INVENTORY STATUS</div><div style="font-size:1.3rem;color:{color};font-weight:800;margin:5px 0 14px">{status}</div><div style="font-size:2.2rem;font-weight:800;color:#111827">{pol["suggested_order_qty"]:,} <span style="font-size:.9rem">units</span></div><div style="color:#64748B;font-size:.82rem">{last.product_name} · suggested order quantity</div></div>',unsafe_allow_html=True)
    st.write("")
    x,y=st.columns(2); x.metric("On hand",pol["on_hand"]); y.metric("Lead-time demand",pol["lead_time_demand"])
    x,y=st.columns(2); x.metric("Safety stock",pol["safety_stock"]); y.metric("Reorder point",pol["reorder_point"])
    x,y=st.columns(2); x.metric("EOQ estimate",f'{pol["eoq"]:.0f}'); y.metric("Lead time",f'{int(last.lead_time_days)} days')
    st.caption(f"Target service level {service:.0%}. Validate supplier minimums, open orders, shelf life, and pack sizes before real use.")

st.markdown('<div class="section-head">Inventory action queue</div><div class="section-sub">All selected products with reorder recommendations</div>',unsafe_allow_html=True)
rows=[]
for (sid,item),g in data.groupby(["store_id","item_id"]):
    if store!="All stores" and sid!=store: continue
    if category!="All categories" and g.category.iloc[0]!=category: continue
    g=g.sort_values("date"); fc=recursive_forecast(model,g,max(horizon,int(g.lead_time_days.iloc[-1])))
    p=inventory_policy(fc.forecast_qty.values,resid_std,int(g.stock_on_hand.iloc[-1]),int(g.lead_time_days.iloc[-1]),
        float(service),float(g.qty_sold.mean()*365),float(g.ordering_cost.iloc[-1]),float(g.unit_cost.iloc[-1]),float(g.holding_cost_rate.iloc[-1]))
    rows.append({"Store":sid,"SKU":item,"Product":g.product_name.iloc[0],"Category":g.category.iloc[0],
        "On hand":p["on_hand"],"Reorder point":p["reorder_point"],"Safety stock":p["safety_stock"],
        "EOQ":round(p["eoq"]),"Suggested order":p["suggested_order_qty"],
        "Status":"Review reorder" if p["reorder_now"] else "Monitor"})
queue=pd.DataFrame(rows).sort_values(["Status","Suggested order"],ascending=[True,False])
st.dataframe(queue,hide_index=True,use_container_width=True)
st.download_button("⬇ Download replenishment recommendations",queue.to_csv(index=False).encode("utf-8"),
    "replenishment_recommendations.csv","text/csv")
with st.expander("Data quality and methodology"):
    st.json(quality)
    st.markdown("This is synthetic data. Stockout-flagged rows remain in the model and may understate latent demand. Residual standard deviation is a simple uncertainty proxy. This starter uses chronological holdout evaluation, not rolling-origin backtesting. EOQ/ROP values are educational estimates, not production purchase orders.")
st.divider()
st.caption("RetailIQ · Portfolio simulation · Not a production ordering system")
