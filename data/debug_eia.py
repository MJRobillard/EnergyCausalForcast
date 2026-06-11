"""
Debug script to find the correct EIA API endpoint and BA code for WAUE.
Run: python data/debug_eia.py
"""
import os, requests, json

KEY = os.environ["EIA_API_KEY"]

def get(url, params=None):
    p = {"api_key": KEY, **(params or {})}
    r = requests.get(url, params=p, timeout=30)
    r.raise_for_status()
    return r.json()

BASE = "https://api.eia.gov/v2/electricity/rto"

# 1. List available sub-routes under /electricity/rto
print("=== /electricity/rto routes ===")
j = get(BASE + "/")
for r in j.get("response", {}).get("routes", []):
    print(" ", r.get("id"), "-", r.get("name"))

# 2. List facet values for respondent in region-data to find WAUE
print("\n=== respondents in /rto/region-data (first 50) ===")
j = get(BASE + "/region-data/facet/respondent")
for item in j.get("response", {}).get("facets", [])[:50]:
    print(f"  {item.get('id'):12s}  {item.get('name')}")

# 3. Try a small data pull for WAUE with no type filter to see what's there
print("\n=== raw data pull for WAUE (first 5 rows, no type filter) ===")
j = get(BASE + "/region-data/data/", {
    "frequency": "hourly",
    "data[0]": "value",
    "facets[respondent][]": "WAUE",
    "sort[0][column]": "period",
    "sort[0][direction]": "desc",
    "length": 5,
})
resp = j.get("response", {})
print("total:", resp.get("total"))
print(json.dumps(resp.get("data", [])[:5], indent=2))

# 4. Also try the interchange/local-data endpoint which some BAs use
print("\n=== /rto/interchange-data respondents containing 'WAU' ===")
try:
    j2 = get(BASE + "/interchange-data/facet/respondent")
    for item in j2.get("response", {}).get("facets", []):
        if "WAU" in item.get("id", "") or "Western" in item.get("name", ""):
            print(f"  {item.get('id'):12s}  {item.get('name')}")
except Exception as e:
    print("  (interchange-data error:", e, ")")
