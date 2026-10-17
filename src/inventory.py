import numpy as np
from scipy.stats import norm

def inventory_policy(forecast, residual_std, on_hand, lead_time_days, service_level=.95,
                     annual_demand=None, ordering_cost=350, unit_cost=100, holding_rate=.20):
    forecast = np.asarray(forecast, dtype=float)
    if len(forecast) == 0: raise ValueError("Forecast must not be empty.")
    lead = max(1, int(lead_time_days))
    lead_values = forecast[:lead]
    if len(lead_values) < lead:
        lead_values = np.pad(lead_values, (0,lead-len(lead_values)), constant_values=float(np.mean(forecast)))
    mean_lead = float(np.maximum(lead_values,0).sum())
    sigma_lead = max(0,float(residual_std))*np.sqrt(lead)
    z = float(norm.ppf(min(max(service_level,.5),.999)))
    safety = z*sigma_lead
    rop = mean_lead+safety
    annual_demand = float(annual_demand if annual_demand is not None else np.mean(forecast)*365)
    holding = float(unit_cost)*float(holding_rate)
    eoq = float(np.sqrt(2*annual_demand*ordering_cost/holding)) if holding > 0 else 0
    qty = max(0,int(np.ceil(max(rop,eoq)-max(0,on_hand))))
    return {"lead_time_demand":round(mean_lead,2),"safety_stock":round(safety,2),
        "reorder_point":round(rop,2),"eoq":round(eoq,2),"on_hand":int(max(0,on_hand)),
        "suggested_order_qty":qty,"reorder_now":bool(on_hand <= rop)}
