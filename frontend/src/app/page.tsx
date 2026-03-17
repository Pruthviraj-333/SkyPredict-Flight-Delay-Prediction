"use client";

import { useState, useEffect, useRef } from "react";
import Navbar from "@/components/Navbar";
import AirportSearch from "@/components/AirportSearch";
import { api, Airline, Airport, PredictionResult, FlightStatusData } from "@/lib/api";

/* ── Animated arc gauge ─────────────────────────── */
function ArcGauge({ value }: { value: number }) {
  const [draw, setDraw] = useState(0);
  useEffect(() => { requestAnimationFrame(() => setDraw(value)); }, [value]);

  const r = 72, cx = 80, cy = 80;
  const startA = (5 / 4) * Math.PI;
  const sweep = (3 / 2) * Math.PI;
  const endA = startA + sweep;
  const total = r * sweep;
  const filled = (draw / 100) * total;

  const arc = (angle: number) => ({
    x: cx + r * Math.cos(angle),
    y: cy + r * Math.sin(angle),
  });
  const s = arc(startA), e = arc(endA);
  const d = `M ${s.x} ${s.y} A ${r} ${r} 0 1 1 ${e.x} ${e.y}`;

  const color = value >= 70 ? "var(--emerald)" : value >= 45 ? "var(--amber)" : "var(--rose)";

  return (
    <svg viewBox="0 0 160 140" width="200" height="175">
      {/* track */}
      <path d={d} fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth="10" strokeLinecap="round" />
      {/* fill */}
      <path
        d={d} fill="none" stroke={color} strokeWidth="10" strokeLinecap="round"
        strokeDasharray={`${total}`}
        strokeDashoffset={total - filled}
        style={{ transition: "stroke-dashoffset 1.2s cubic-bezier(0.4,0,0.2,1), stroke 0.4s" }}
      />
      {/* value */}
      <text x={cx} y={cy - 2} textAnchor="middle" fill="var(--text-primary)"
        fontFamily="Space Grotesk" fontSize="38" fontWeight="700">{Math.round(draw)}<tspan fontSize="18">%</tspan></text>
      <text x={cx} y={cy + 18} textAnchor="middle" fill="var(--text-secondary)" fontSize="11" fontWeight="500">on-time</text>
    </svg>
  );
}

/* ── Main page ──────────────────────────────────── */
export default function Home() {
  const [airlines, setAirlines] = useState<Airline[]>([]);
  const [airports, setAirports] = useState<Airport[]>([]);
  const [dataReady, setDataReady] = useState(false);
  const [carrier, setCarrier] = useState("");
  const [origin, setOrigin] = useState("");
  const [dest, setDest] = useState("");
  const [date, setDate] = useState("");
  const [depTime, setDepTime] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [err, setErr] = useState("");
  const resultRef = useRef<HTMLDivElement>(null);
  const [flightNum, setFlightNum] = useState("");
  const [trackLoading, setTrackLoading] = useState(false);
  const [flightStatus, setFlightStatus] = useState<FlightStatusData | null>(null);
  const [trackErr, setTrackErr] = useState("");

  useEffect(() => {
    Promise.all([
      api.getAirlines().then(setAirlines).catch(() => { }),
      api.getAirports().then(setAirports).catch(() => { }),
    ]).finally(() => setDataReady(true));
    setDate(new Date().toISOString().split("T")[0]);
  }, []);

  const today = new Date().toISOString().split("T")[0];

  const predict = async () => {
    // Field-level validation with specific messages
    if (!carrier) { setErr("Please select an airline."); return; }
    if (!origin) { setErr("Please select a departure airport."); return; }
    if (!dest) { setErr("Please select a destination airport."); return; }
    if (origin === dest) { setErr("Origin and destination cannot be the same airport."); return; }
    if (!date) { setErr("Please select a travel date."); return; }
    if (!depTime) { setErr("Please set a departure time."); return; }
    setLoading(true); setErr(""); setResult(null);
    try {
      const t = depTime.replace(":", "").padStart(4, "0");
      const p = await api.predict({ carrier, origin: origin.toUpperCase(), dest: dest.toUpperCase(), date, dep_time: t });
      setResult(p);
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "center" }), 100);
    } catch (e: any) {
      const msg = e.message || "";
      if (msg.includes("400")) setErr("Invalid flight details. Please check your inputs and try again.");
      else if (msg.includes("500")) setErr("Server error — please try again in a moment.");
      else if (msg.includes("fetch") || msg.includes("network")) setErr("Cannot connect to server. Is the backend running?");
      else setErr(msg || "Something went wrong. Please try again.");
    }
    finally { setLoading(false); }
  };

  const riskClass = result ? `risk-${result.risk_level.toLowerCase()}` : "";
  const onTimePct = result ? result.on_time_probability * 100 : 0;
  const delayPct = result ? result.delay_probability * 100 : 0;

  const trackFlight = async () => {
    const num = flightNum.trim();
    if (!num) { setTrackErr("Enter a flight number (e.g. AA100)."); return; }
    if (num.length < 3) { setTrackErr("Flight number must be at least 3 characters (e.g. AA100)."); return; }
    setTrackLoading(true); setTrackErr(""); setFlightStatus(null);
    try {
      const status = await api.getFlightStatus(num);
      setFlightStatus(status);
    } catch (e: any) {
      const msg = e.message || "";
      if (msg.includes("404")) setTrackErr(`Flight "${num}" not found. Check the flight number and try again.`);
      else if (msg.includes("503")) setTrackErr("Flight tracking is temporarily unavailable. Please try again later.");
      else if (msg.includes("fetch") || msg.includes("network")) setTrackErr("Cannot connect to server. Is the backend running?");
      else setTrackErr(msg || "Could not fetch flight status. Please try again.");
    }
    finally { setTrackLoading(false); }
  };

  const statusColor = (s: string) => {
    switch (s) {
      case "active": return { bg: "var(--sky-muted)", color: "var(--sky)", label: "In Flight" };
      case "landed": return { bg: "var(--emerald-muted)", color: "var(--emerald)", label: "Landed" };
      case "scheduled": return { bg: "var(--amber-muted)", color: "var(--amber)", label: "Scheduled" };
      case "cancelled": return { bg: "var(--rose-muted)", color: "var(--rose)", label: "Cancelled" };
      case "diverted": return { bg: "var(--rose-muted)", color: "var(--rose)", label: "Diverted" };
      case "incident": return { bg: "var(--rose-muted)", color: "var(--rose)", label: "Incident" };
      default: return { bg: "rgba(255,255,255,0.06)", color: "var(--text-secondary)", label: s || "Unknown" };
    }
  };

  const formatTime = (t: string | null) => {
    if (!t) return "--";
    try { return new Date(t).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }); }
    catch { return t; }
  };

  return (
    <>
      <Navbar />
      <div className="page">
        {/* ── HEADER ── */}
        <div className="animate-enter" style={{ marginBottom: 36 }}>
          <p style={{ fontSize: 12, fontWeight: 600, color: "var(--sky)", textTransform: "uppercase", letterSpacing: 1.2, marginBottom: 6 }}>
            Flight Delay Predictor
          </p>
          <h1 style={{ fontSize: 30, fontWeight: 700, letterSpacing: -0.5, lineHeight: 1.2, color: "var(--text-primary)" }}>
            Check if your flight will<br />arrive on time
          </h1>
          <p style={{ marginTop: 8, fontSize: 14.5, color: "var(--text-secondary)", maxWidth: 440 }}>
            Powered by machine learning trained on 600,000+ real U.S. domestic flights.
          </p>
        </div>

        {/* ── FORM ── */}
        <div className="card animate-enter-d1" style={{ padding: "28px 28px 24px", marginBottom: 24 }}>
          {!dataReady && (
            <div style={{ textAlign: "center", padding: "20px 0", color: "var(--text-secondary)", fontSize: 13 }}>
              <span className="spin" style={{ marginRight: 8 }} />Loading airlines and airports…
            </div>
          )}
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 14, marginBottom: 14 }}>
            <div>
              <label className="field-label">Airline</label>
              <select className="field-input" value={carrier} onChange={e => setCarrier(e.target.value)}>
                <option value="">Choose airline</option>
                {airlines.map(a => <option key={a.code} value={a.code}>{a.name} ({a.code})</option>)}
              </select>
            </div>
            <div>
              <label className="field-label">From</label>
              <AirportSearch
                airports={airports}
                value={origin}
                onChange={setOrigin}
                placeholder="Type city or code"
                id="origin"
              />
            </div>
            <div>
              <label className="field-label">To</label>
              <AirportSearch
                airports={airports}
                value={dest}
                onChange={setDest}
                placeholder="Type city or code"
                id="dest"
              />
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))", gap: 14, alignItems: "end" }}>
            <div>
              <label className="field-label">Date</label>
              <input className="field-input" type="date" value={date} min={today} onChange={e => setDate(e.target.value)} />
            </div>
            <div>
              <label className="field-label">Departure</label>
              <input className="field-input" type="time" value={depTime} onChange={e => setDepTime(e.target.value)}
                onKeyDown={e => e.key === "Enter" && predict()} />
            </div>
            <button className="btn btn-sky" onClick={predict} disabled={loading || !dataReady} style={{ height: 42, minWidth: 150 }}>
              {loading ? <><span className="spin" /> Checking…</> : "Check delay"}
            </button>
          </div>

          {err && (
            <p style={{ marginTop: 12, fontSize: 13, color: "var(--rose)", background: "var(--rose-muted)", padding: "8px 14px", borderRadius: 8 }}>
              {err}
            </p>
          )}
        </div>

        {/* ── RESULT ── */}
        {result && (
          <div ref={resultRef} className="card animate-enter" style={{ padding: 28 }}>
            {/* Route header */}
            <div style={{ display: "flex", alignItems: "center", gap: 16, marginBottom: 24 }}>
              <div style={{ flex: 1 }}>
                <div style={{ display: "flex", alignItems: "baseline", gap: 10, marginBottom: 2 }}>
                  <span style={{ fontFamily: "Space Grotesk", fontSize: 22, fontWeight: 700 }}>{result.origin}</span>
                  <svg width="28" height="12" viewBox="0 0 28 12" fill="none">
                    <line x1="0" y1="6" x2="22" y2="6" stroke="var(--text-dim)" strokeWidth="1.5" strokeDasharray="3 3" />
                    <path d="M20 2l6 4-6 4" stroke="var(--sky)" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />
                  </svg>
                  <span style={{ fontFamily: "Space Grotesk", fontSize: 22, fontWeight: 700 }}>{result.dest}</span>
                </div>
                <p style={{ fontSize: 13, color: "var(--text-secondary)" }}>
                  {result.carrier} · {result.date} · {result.dep_time.slice(0, 2)}:{result.dep_time.slice(2)}
                  {result.distance_miles > 0 && ` · ${Math.round(result.distance_miles).toLocaleString()} mi`}
                  {result.flight_duration_minutes > 0 && ` · ${Math.floor(result.flight_duration_minutes / 60)}h ${Math.round(result.flight_duration_minutes % 60)}m`}
                </p>
              </div>
              <span className={`risk-tag ${riskClass}`}>
                {result.risk_level} risk
              </span>
            </div>

            {/* Gauge + bars */}
            <div style={{ display: "flex", gap: 40, alignItems: "center", flexWrap: "wrap" }}>
              <ArcGauge value={onTimePct} />

              <div style={{ flex: 1, minWidth: 260 }}>
                {/* On-time bar */}
                <div style={{ marginBottom: 18 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
                    <span style={{ fontSize: 13, color: "var(--emerald)", fontWeight: 600 }}>On-time</span>
                    <span style={{ fontSize: 13, fontFamily: "Space Grotesk", fontWeight: 700 }}>{onTimePct.toFixed(1)}%</span>
                  </div>
                  <div className="progress-track">
                    <div className="progress-fill" style={{ width: `${onTimePct}%`, background: "var(--emerald)" }} />
                  </div>
                </div>
                {/* Delay bar */}
                <div style={{ marginBottom: 24 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 5 }}>
                    <span style={{ fontSize: 13, color: "var(--rose)", fontWeight: 600 }}>Delayed</span>
                    <span style={{ fontSize: 13, fontFamily: "Space Grotesk", fontWeight: 700 }}>{delayPct.toFixed(1)}%</span>
                  </div>
                  <div className="progress-track">
                    <div className="progress-fill" style={{ width: `${delayPct}%`, background: "var(--rose)" }} />
                  </div>
                </div>

                {/* Verdict */}
                <div style={{
                  padding: "12px 16px", borderRadius: 10, fontSize: 13.5, lineHeight: 1.6,
                  background: result.is_delayed ? "var(--rose-muted)" : "var(--emerald-muted)",
                  color: result.is_delayed ? "var(--rose)" : "var(--emerald)",
                  border: `1px solid ${result.is_delayed ? "rgba(251,113,133,0.15)" : "rgba(52,211,153,0.15)"}`,
                }}>
                  {result.is_delayed
                    ? "This flight has a high probability of arriving late. Consider checking alternatives or arriving early at the airport."
                    : "This flight has a strong on-time track record for this route, carrier, and time slot. Have a great flight!"
                  }
                </div>

                {/* Model indicator */}
                <div style={{
                  display: "flex", alignItems: "center", gap: 8,
                  marginTop: 14, fontSize: 12, color: "var(--text-dim)",
                }}>
                  <span style={{
                    display: "inline-flex", alignItems: "center", gap: 4,
                    padding: "3px 10px", borderRadius: 5, fontSize: 11, fontWeight: 600,
                    background: result.model_used === "primary" ? "var(--sky-muted)" : "rgba(255,255,255,0.04)",
                    color: result.model_used === "primary" ? "var(--sky)" : "var(--text-dim)",
                  }}>
                    {result.model_used === "primary" ? "☁ Primary Model (Weather)" : "⚡ Fallback Model"}
                  </span>
                  {result.weather_available && (
                    <span style={{ fontSize: 11, color: "var(--emerald)" }}>✓ Live weather data</span>
                  )}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <p style={{ textAlign: "center", color: "var(--text-dim)", fontSize: 11.5, marginTop: 40 }}>
          Dual-Model: XGBoost Primary (weather) + Fallback · BTS Oct 2025 · 601K flights
        </p>

        {/* ── LIVE FLIGHT TRACKING ── */}
        <div style={{ marginTop: 48, borderTop: "1px solid var(--border)", paddingTop: 40 }}>
          <div className="animate-enter" style={{ marginBottom: 24 }}>
            <p style={{ fontSize: 12, fontWeight: 600, color: "var(--amber)", textTransform: "uppercase", letterSpacing: 1.2, marginBottom: 6 }}>
              Live Flight Tracking
            </p>
            <h2 style={{ fontSize: 24, fontWeight: 700, letterSpacing: -0.3, color: "var(--text-primary)" }}>
              Track a flight in real-time
            </h2>
            <p style={{ marginTop: 6, fontSize: 13.5, color: "var(--text-secondary)" }}>
              Enter a flight number to see its live status from AviationStack.
            </p>
          </div>

          <div className="card animate-enter-d1" style={{ padding: "24px 28px" }}>
            <div style={{ display: "flex", gap: 12, alignItems: "end" }}>
              <div style={{ flex: 1, maxWidth: 260 }}>
                <label className="field-label">Flight Number</label>
                <input
                  className="field-input"
                  placeholder="e.g. AA100, DL402, UA1234"
                  value={flightNum}
                  onChange={e => setFlightNum(e.target.value.toUpperCase())}
                  onKeyDown={e => e.key === "Enter" && trackFlight()}
                  maxLength={10}
                />
              </div>
              <button className="btn btn-sky" onClick={trackFlight} disabled={trackLoading} style={{ height: 42, minWidth: 120 }}>
                {trackLoading ? <><span className="spin" /> Tracking…</> : "Track"}
              </button>
            </div>

            {trackErr && (
              <p style={{ marginTop: 12, fontSize: 13, color: "var(--rose)", background: "var(--rose-muted)", padding: "8px 14px", borderRadius: 8 }}>
                {trackErr}
              </p>
            )}
          </div>

          {/* Flight Status Result */}
          {flightStatus && (() => {
            const sc = statusColor(flightStatus.status);
            return (
              <div className="card animate-enter" style={{ padding: 28, marginTop: 16 }}>
                {/* Header */}
                <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", marginBottom: 24 }}>
                  <div>
                    <div style={{ fontFamily: "Space Grotesk", fontSize: 22, fontWeight: 700 }}>
                      {flightStatus.flight_iata}
                    </div>
                    <p style={{ fontSize: 13, color: "var(--text-secondary)" }}>{flightStatus.airline_name}</p>
                  </div>
                  <span style={{
                    padding: "5px 14px", borderRadius: 6, fontSize: 12, fontWeight: 700,
                    textTransform: "uppercase", letterSpacing: 0.4,
                    background: sc.bg, color: sc.color,
                  }}>
                    {sc.label}
                  </span>
                </div>

                {/* Departure / Arrival */}
                <div style={{ display: "grid", gridTemplateColumns: "1fr auto 1fr", gap: 16, alignItems: "center" }}>
                  {/* Departure */}
                  <div>
                    <div style={{ fontSize: 11, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 4 }}>Departure</div>
                    <div style={{ fontFamily: "Space Grotesk", fontSize: 20, fontWeight: 700 }}>
                      {flightStatus.departure.airport}
                    </div>
                    <div style={{ fontSize: 13, color: "var(--text-secondary)", marginTop: 4 }}>
                      Scheduled: {formatTime(flightStatus.departure.scheduled)}
                    </div>
                    {flightStatus.departure.actual && (
                      <div style={{ fontSize: 13, color: "var(--text-secondary)" }}>
                        Actual: {formatTime(flightStatus.departure.actual)}
                      </div>
                    )}
                    {flightStatus.departure.delay_minutes != null && flightStatus.departure.delay_minutes > 0 && (
                      <div style={{ fontSize: 13, fontWeight: 600, color: "var(--rose)", marginTop: 4 }}>
                        +{flightStatus.departure.delay_minutes} min delay
                      </div>
                    )}
                  </div>

                  {/* Arrow */}
                  <div style={{ textAlign: "center" }}>
                    <svg width="60" height="20" viewBox="0 0 60 20">
                      <line x1="0" y1="10" x2="48" y2="10" stroke="var(--text-dim)" strokeWidth="1.5" strokeDasharray="4 3" />
                      <path d="M46 5l8 5-8 5" stroke="var(--sky)" strokeWidth="1.5" fill="none" strokeLinecap="round" />
                    </svg>
                  </div>

                  {/* Arrival */}
                  <div style={{ textAlign: "right" }}>
                    <div style={{ fontSize: 11, color: "var(--text-dim)", textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 4 }}>Arrival</div>
                    <div style={{ fontFamily: "Space Grotesk", fontSize: 20, fontWeight: 700 }}>
                      {flightStatus.arrival.airport}
                    </div>
                    <div style={{ fontSize: 13, color: "var(--text-secondary)", marginTop: 4 }}>
                      Scheduled: {formatTime(flightStatus.arrival.scheduled)}
                    </div>
                    {flightStatus.arrival.estimated && (
                      <div style={{ fontSize: 13, color: "var(--text-secondary)" }}>
                        Estimated: {formatTime(flightStatus.arrival.estimated)}
                      </div>
                    )}
                    {flightStatus.arrival.delay_minutes != null && flightStatus.arrival.delay_minutes > 0 && (
                      <div style={{ fontSize: 13, fontWeight: 600, color: "var(--rose)", marginTop: 4 }}>
                        +{flightStatus.arrival.delay_minutes} min delay
                      </div>
                    )}
                  </div>
                </div>

                {/* Live position */}
                {flightStatus.live && (
                  <div style={{
                    marginTop: 20, padding: "14px 16px", borderRadius: 10,
                    background: "var(--sky-muted)", border: "1px solid rgba(56,189,248,0.12)",
                  }}>
                    <div style={{ fontSize: 11, color: "var(--sky)", fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.5, marginBottom: 6 }}>
                      Live Position
                    </div>
                    <div style={{ display: "flex", gap: 24, fontSize: 13, color: "var(--text-secondary)" }}>
                      <span>Lat: <strong style={{ color: "var(--text-primary)" }}>{flightStatus.live.latitude.toFixed(2)}°</strong></span>
                      <span>Lon: <strong style={{ color: "var(--text-primary)" }}>{flightStatus.live.longitude.toFixed(2)}°</strong></span>
                      <span>Alt: <strong style={{ color: "var(--text-primary)" }}>{Math.round(flightStatus.live.altitude).toLocaleString()} ft</strong></span>
                      <span>Speed: <strong style={{ color: "var(--text-primary)" }}>{Math.round(flightStatus.live.speed_horizontal)} km/h</strong></span>
                    </div>
                  </div>
                )}
              </div>
            );
          })()}
        </div>
      </div>
    </>
  );
}
