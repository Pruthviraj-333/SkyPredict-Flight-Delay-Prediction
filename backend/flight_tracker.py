"""
Flight Tracker Service — Real-time flight status using AviationStack API.

Provides live flight status (scheduled, active, landed, cancelled, delayed)
alongside our ML predictions.
"""

import os
import requests
from typing import Optional
from datetime import datetime


class FlightStatus:
    """Structured flight status result."""
    def __init__(self, flight_iata, airline_name, departure, arrival,
                 status, dep_delay, arr_delay, dep_actual, arr_estimated,
                 dep_airport, arr_airport, live_data):
        self.flight_iata = flight_iata
        self.airline_name = airline_name
        self.departure = departure
        self.arrival = arrival
        self.status = status
        self.dep_delay = dep_delay
        self.arr_delay = arr_delay
        self.dep_actual = dep_actual
        self.arr_estimated = arr_estimated
        self.dep_airport = dep_airport
        self.arr_airport = arr_airport
        self.live_data = live_data

    def to_dict(self):
        return {
            "flight_iata": self.flight_iata,
            "airline_name": self.airline_name,
            "status": self.status,
            "departure": {
                "airport": self.dep_airport,
                "scheduled": self.departure,
                "actual": self.dep_actual,
                "delay_minutes": self.dep_delay,
            },
            "arrival": {
                "airport": self.arr_airport,
                "scheduled": self.arrival,
                "estimated": self.arr_estimated,
                "delay_minutes": self.arr_delay,
            },
            "live": self.live_data,
        }


class FlightTracker:
    """Service to fetch real-time flight status from AviationStack."""

    BASE_URL = "http://api.aviationstack.com/v1"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("AVIATIONSTACK_API_KEY", "")
        if not self.api_key:
            print("[WARN] No AviationStack API key set. Flight tracking disabled.")

    def is_available(self) -> bool:
        return bool(self.api_key)

    def get_flight_status(self, flight_iata: str) -> Optional[dict]:
        """
        Fetch live status for a flight by its IATA code (e.g., AA100, DL402).
        Returns structured flight info or None if not found.
        """
        if not self.api_key:
            return None

        flight_iata = flight_iata.upper().strip().replace(" ", "")

        try:
            params = {
                "access_key": self.api_key,
                "flight_iata": flight_iata,
            }
            resp = requests.get(f"{self.BASE_URL}/flights", params=params, timeout=10)

            if resp.status_code != 200:
                return {"error": f"API returned status {resp.status_code}"}

            data = resp.json()

            if "error" in data:
                return {"error": data["error"].get("message", "API error")}

            flights = data.get("data", [])
            if not flights:
                return {"error": f"No flight found for {flight_iata}"}

            # Get the most relevant flight (first result)
            flight = flights[0]

            departure = flight.get("departure", {})
            arrival = flight.get("arrival", {})
            airline = flight.get("airline", {})
            live = flight.get("live", None)

            # Build live data if available
            live_data = None
            if live and live.get("latitude"):
                live_data = {
                    "latitude": live.get("latitude"),
                    "longitude": live.get("longitude"),
                    "altitude": live.get("altitude"),
                    "speed_horizontal": live.get("speed_horizontal"),
                    "is_ground": live.get("is_ground", False),
                    "updated": live.get("updated"),
                }

            status = FlightStatus(
                flight_iata=flight.get("flight", {}).get("iata", flight_iata),
                airline_name=airline.get("name", "Unknown"),
                departure=departure.get("scheduled"),
                arrival=arrival.get("scheduled"),
                status=flight.get("flight_status", "unknown"),
                dep_delay=departure.get("delay"),
                arr_delay=arrival.get("delay"),
                dep_actual=departure.get("actual"),
                arr_estimated=arrival.get("estimated"),
                dep_airport=departure.get("airport", "Unknown"),
                arr_airport=arrival.get("airport", "Unknown"),
                live_data=live_data,
            )

            return status.to_dict()

        except requests.Timeout:
            return {"error": "AviationStack API timeout"}
        except requests.ConnectionError:
            return {"error": "Could not connect to AviationStack"}
        except Exception as e:
            return {"error": str(e)}
