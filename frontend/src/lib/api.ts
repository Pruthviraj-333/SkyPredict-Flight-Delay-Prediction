const API_BASE = process.env.NEXT_PUBLIC_API_URL || 'http://127.0.0.1:8000';

export interface Airline {
    code: string;
    name: string;
}

export interface Airport {
    code: string;
    name: string;
}

export interface PredictionResult {
    carrier: string;
    origin: string;
    dest: string;
    date: string;
    dep_time: string;
    is_delayed: boolean;
    delay_probability: number;
    on_time_probability: number;
    risk_level: string;
    distance_miles: number;
    flight_duration_minutes: number;
    model_used: string;
    weather_available: boolean;
    predicted_delay_minutes: number;
}

export interface Stats {
    overall_delay_rate: number;
    total_airlines: number;
    total_airports: number;
    total_routes: number;
    worst_carrier: string;
    worst_carrier_rate: number;
    worst_route: string;
    worst_route_rate: number;
    best_departure_hour: number;
    best_hour_rate: number;
}

export interface TrendData {
    day: string;
    day_num: number;
    delay_rate: number;
}

export interface RouteData {
    route: string;
    delay_rate: number;
}

export interface CarrierData {
    code: string;
    name: string;
    delay_rate: number;
}

export interface HourData {
    hour: number;
    label: string;
    delay_rate: number;
}

async function fetchJSON(url: string, options?: RequestInit) {
    const res = await fetch(`${API_BASE}${url}`, {
        ...options,
        headers: { 'Content-Type': 'application/json', ...options?.headers },
    });
    if (!res.ok) throw new Error(`API error: ${res.status}`);
    return res.json();
}

export const api = {
    health: () => fetchJSON('/api/health'),

    getAirlines: async (): Promise<Airline[]> => {
        const data = await fetchJSON('/api/airlines');
        return data.airlines;
    },

    getAirports: async (): Promise<Airport[]> => {
        const data = await fetchJSON('/api/airports');
        return data.airports;
    },

    predict: async (flight: {
        carrier: string; origin: string; dest: string;
        date: string; dep_time: string;
    }): Promise<PredictionResult> => {
        const data = await fetchJSON('/api/predict', {
            method: 'POST',
            body: JSON.stringify(flight),
        });
        return data.prediction;
    },

    getStats: async (): Promise<Stats> => {
        const data = await fetchJSON('/api/stats');
        return data.stats;
    },

    getTrends: async (): Promise<TrendData[]> => {
        const data = await fetchJSON('/api/analytics/trends');
        return data.trends;
    },

    getRoutes: async (topN = 10): Promise<RouteData[]> => {
        const data = await fetchJSON(`/api/analytics/routes?top_n=${topN}`);
        return data.routes;
    },

    getCarriers: async (): Promise<CarrierData[]> => {
        const data = await fetchJSON('/api/analytics/carriers');
        return data.carriers;
    },

    getHours: async (): Promise<HourData[]> => {
        const data = await fetchJSON('/api/analytics/hours');
        return data.hours;
    },

    getHeatmap: async () => {
        const data = await fetchJSON('/api/analytics/heatmap');
        return data.heatmap;
    },

    getFlightStatus: async (flightIata: string): Promise<FlightStatusData> => {
        const data = await fetchJSON(`/api/flight-status/${flightIata}`);
        return data.flight_status;
    },
};

export interface FlightStatusData {
    flight_iata: string;
    airline_name: string;
    status: string;
    departure: {
        airport: string;
        scheduled: string | null;
        actual: string | null;
        delay_minutes: number | null;
    };
    arrival: {
        airport: string;
        scheduled: string | null;
        estimated: string | null;
        delay_minutes: number | null;
    };
    live: {
        latitude: number;
        longitude: number;
        altitude: number;
        speed_horizontal: number;
        is_ground: boolean;
        updated: string;
    } | null;
}
