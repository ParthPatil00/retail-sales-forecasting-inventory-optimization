import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.inventory import inventory_policy

def test_inventory_policy_nonnegative():
    result=inventory_policy([10,12,11,13],2,0,2,.95,4000,100,20,.2)
    assert result["safety_stock"] >= 0
    assert result["reorder_point"] >= result["lead_time_demand"]
    assert result["suggested_order_qty"] >= 0
    assert result["reorder_now"]
