"""Quick test for flight tracking endpoint."""
import requests
import json

API = "http://127.0.0.1:8000"

print("=" * 50)
print("  FLIGHT TRACKING TEST")
print("=" * 50)

# Test health (check tracking enabled)
r = requests.get(f"{API}/api/health", timeout=10)
health = r.json()
print(f"\n  Flight tracking: {'enabled' if health.get('flight_tracking') else 'disabled'}")

# Test flight status
print("\n  Testing flight status for AA100...")
try:
    r = requests.get(f"{API}/api/flight-status/AA100", timeout=15)
    if r.status_code == 200:
        data = r.json()["flight_status"]
        print(f"  Status: {r.status_code} OK")
        print(f"  Flight: {data['flight_iata']}")
        print(f"  Airline: {data['airline_name']}")
        print(f"  Status: {data['status']}")
        print(f"  From: {data['departure']['airport']}")
        print(f"  To: {data['arrival']['airport']}")
        if data['departure'].get('delay_minutes'):
            print(f"  Dep delay: {data['departure']['delay_minutes']} min")
        if data.get('live'):
            print(f"  Live: {data['live']['latitude']}, {data['live']['longitude']}")
        print(f"\n  Full response:")
        print(f"  {json.dumps(data, indent=2)}")
    else:
        print(f"  Status: {r.status_code}")
        print(f"  Response: {r.text[:300]}")
except Exception as e:
    print(f"  Error: {e}")

print(f"\n{'=' * 50}")
