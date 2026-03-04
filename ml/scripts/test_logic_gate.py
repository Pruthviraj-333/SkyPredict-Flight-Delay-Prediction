"""Test the dual-model logic gate."""
import requests
import json

API = "http://127.0.0.1:8000"

print("=" * 55)
print("  LOGIC GATE TEST")
print("=" * 55)

# Health
h = requests.get(f"{API}/api/health").json()
print(f"\n  Health: {h['status']}")
print(f"  Flight tracking: {h.get('flight_tracking')}")

# Test 1: Today (should get weather → primary model)
print(f"\n{'─' * 55}")
print("  Test 1: Today's flight (weather should be available)")
print(f"{'─' * 55}")
r = requests.post(f"{API}/api/predict", json={
    "carrier": "AA", "origin": "JFK", "dest": "LAX",
    "date": "2026-03-04", "dep_time": "0800"
}).json()
p = r["prediction"]
print(f"  Route:    {p['origin']} → {p['dest']}")
print(f"  Delay:    {p['delay_probability']*100:.1f}%")
print(f"  Risk:     {p['risk_level']}")
print(f"  Model:    {p['model_used']}")
print(f"  Weather:  {p['weather_available']}")

# Test 2: Far future (no weather → fallback)
print(f"\n{'─' * 55}")
print("  Test 2: Far future (weather should NOT be available)")
print(f"{'─' * 55}")
r2 = requests.post(f"{API}/api/predict", json={
    "carrier": "DL", "origin": "ATL", "dest": "ORD",
    "date": "2027-06-15", "dep_time": "1400"
}).json()
p2 = r2["prediction"]
print(f"  Route:    {p2['origin']} → {p2['dest']}")
print(f"  Delay:    {p2['delay_probability']*100:.1f}%")
print(f"  Risk:     {p2['risk_level']}")
print(f"  Model:    {p2['model_used']}")
print(f"  Weather:  {p2['weather_available']}")

# Summary
print(f"\n{'=' * 55}")
if p["model_used"] == "primary" and p2["model_used"] == "fallback":
    print("  ✅ LOGIC GATE WORKING CORRECTLY!")
    print("     Today → Primary (weather), Future → Fallback (no weather)")
elif p["model_used"] == "primary" and p2["model_used"] == "primary":
    print("  ⚠  Both used primary (Open-Meteo may have forecast data)")
else:
    print(f"  Results: Test1={p['model_used']}, Test2={p2['model_used']}")
print(f"{'=' * 55}")
