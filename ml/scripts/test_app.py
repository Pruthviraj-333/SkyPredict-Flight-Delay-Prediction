"""Quick test for all API endpoints and frontend."""
import requests
import json

API = "http://127.0.0.1:8000"
FRONTEND = "http://127.0.0.1:3000"

print("=" * 60)
print("  SKYPREDICT — FULL STACK VERIFICATION")
print("=" * 60)

# Test backend endpoints
endpoints = [
    ("GET", "/api/health"),
    ("GET", "/api/airlines"),
    ("GET", "/api/airports"),
    ("GET", "/api/stats"),
    ("GET", "/api/analytics/trends"),
    ("GET", "/api/analytics/routes"),
    ("GET", "/api/analytics/carriers"),
    ("GET", "/api/analytics/hours"),
    ("GET", "/api/analytics/heatmap"),
]

print("\n[BACKEND API TESTS]")
all_ok = True
for method, path in endpoints:
    try:
        r = requests.get(f"{API}{path}", timeout=10)
        status = "✓" if r.status_code == 200 else "✗"
        if r.status_code != 200:
            all_ok = False
        print(f"  {status} {method} {path:30s} -> {r.status_code}")
    except Exception as e:
        print(f"  ✗ {method} {path:30s} -> ERROR: {e}")
        all_ok = False

# Test prediction endpoint
print("\n[PREDICTION TEST]")
try:
    r = requests.post(f"{API}/api/predict", json={
        "carrier": "DL", "origin": "ATL", "dest": "ORD",
        "date": "2026-02-22", "dep_time": "1800"
    }, timeout=10)
    if r.status_code == 200:
        pred = r.json()["prediction"]
        print(f"  ✓ POST /api/predict            -> {r.status_code}")
        print(f"    Route: {pred['origin']} -> {pred['dest']}")
        print(f"    Delay prob: {pred['delay_probability']*100:.1f}%")
        print(f"    Risk: {pred['risk_level']}")
        print(f"    Distance: {pred['distance_miles']} mi")
    else:
        print(f"  ✗ POST /api/predict -> {r.status_code}: {r.text[:200]}")
        all_ok = False
except Exception as e:
    print(f"  ✗ POST /api/predict -> ERROR: {e}")
    all_ok = False

# Test frontend
print("\n[FRONTEND TEST]")
try:
    r = requests.get(FRONTEND, timeout=10)
    has_sky = "SkyPredict" in r.text
    print(f"  {'✓' if r.status_code == 200 else '✗'} GET / -> {r.status_code} ({len(r.text)} bytes)")
    print(f"  {'✓' if has_sky else '✗'} Contains 'SkyPredict' branding: {has_sky}")
except Exception as e:
    print(f"  ✗ Frontend not reachable: {e}")
    all_ok = False

try:
    r = requests.get(f"{FRONTEND}/staff", timeout=10)
    has_staff = "Staff" in r.text or "Analytics" in r.text
    print(f"  {'✓' if r.status_code == 200 else '✗'} GET /staff -> {r.status_code}")
except Exception as e:
    print(f"  ✗ Staff page not reachable: {e}")

print(f"\n{'=' * 60}")
if all_ok:
    print("  ✅ ALL TESTS PASSED")
else:
    print("  ⚠️  SOME TESTS FAILED — check output above")
print(f"{'=' * 60}")
print(f"\n  Frontend: http://localhost:3000")
print(f"  Staff:    http://localhost:3000/staff")
print(f"  API Docs: http://localhost:8000/docs")
