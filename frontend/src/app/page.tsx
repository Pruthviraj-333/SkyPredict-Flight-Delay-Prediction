"use client";

import { useState, useEffect, useRef } from "react";
import Navbar from "@/components/Navbar";
import { api, Airline, PredictionResult } from "@/lib/api";

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
  const [airports, setAirports] = useState<string[]>([]);
  const [carrier, setCarrier] = useState("");
  const [origin, setOrigin] = useState("");
  const [dest, setDest] = useState("");
  const [date, setDate] = useState("");
  const [depTime, setDepTime] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [err, setErr] = useState("");
  const resultRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    api.getAirlines().then(setAirlines).catch(() => { });
    api.getAirports().then(setAirports).catch(() => { });
    setDate(new Date().toISOString().split("T")[0]);
  }, []);

  const predict = async () => {
    if (!carrier || !origin || !dest || !date || !depTime) { setErr("Fill in all fields."); return; }
    setLoading(true); setErr(""); setResult(null);
    try {
      const t = depTime.replace(":", "").padStart(4, "0");
      const p = await api.predict({ carrier, origin, dest, date, dep_time: t });
      setResult(p);
      setTimeout(() => resultRef.current?.scrollIntoView({ behavior: "smooth", block: "center" }), 100);
    } catch (e: any) { setErr(e.message || "Something went wrong."); }
    finally { setLoading(false); }
  };

  const riskClass = result ? `risk-${result.risk_level.toLowerCase()}` : "";
  const onTimePct = result ? result.on_time_probability * 100 : 0;
  const delayPct = result ? result.delay_probability * 100 : 0;

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
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 14, marginBottom: 14 }}>
            <div>
              <label className="field-label">Airline</label>
              <select className="field-input" value={carrier} onChange={e => setCarrier(e.target.value)}>
                <option value="">Choose airline</option>
                {airlines.map(a => <option key={a.code} value={a.code}>{a.name} ({a.code})</option>)}
              </select>
            </div>
            <div>
              <label className="field-label">From</label>
              <input className="field-input" list="ol" placeholder="e.g. JFK" value={origin}
                onChange={e => setOrigin(e.target.value.toUpperCase())} maxLength={4} />
              <datalist id="ol">{airports.map(a => <option key={a} value={a} />)}</datalist>
            </div>
            <div>
              <label className="field-label">To</label>
              <input className="field-input" list="dl" placeholder="e.g. LAX" value={dest}
                onChange={e => setDest(e.target.value.toUpperCase())} maxLength={4} />
              <datalist id="dl">{airports.map(a => <option key={a} value={a} />)}</datalist>
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr auto", gap: 14, alignItems: "end" }}>
            <div>
              <label className="field-label">Date</label>
              <input className="field-input" type="date" value={date} onChange={e => setDate(e.target.value)} />
            </div>
            <div>
              <label className="field-label">Departure</label>
              <input className="field-input" type="time" value={depTime} onChange={e => setDepTime(e.target.value)} />
            </div>
            <button className="btn btn-sky" onClick={predict} disabled={loading} style={{ height: 42, minWidth: 150 }}>
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
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <p style={{ textAlign: "center", color: "var(--text-dim)", fontSize: 11.5, marginTop: 40 }}>
          Model: XGBoost · Trained on BTS Oct 2025 · 601K flights · 72.5% test accuracy
        </p>
      </div>
    </>
  );
}
