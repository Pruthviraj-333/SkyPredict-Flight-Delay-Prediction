"use client";

import { useState, useEffect } from "react";
import Navbar from "@/components/Navbar";
import AuthGuard from "@/components/AuthGuard";
import { api, Stats, TrendData, RouteData, CarrierData, HourData } from "@/lib/api";
import {
    BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
    AreaChart, Area, CartesianGrid, Cell,
} from "recharts";

/* ── Custom tooltip ──────────────────────── */
function Tip({ active, payload, label }: any) {
    if (!active || !payload?.length) return null;
    return (
        <div style={{
            background: "#111827", border: "1px solid rgba(56,189,248,0.15)",
            borderRadius: 8, padding: "8px 12px", fontSize: 12.5,
        }}>
            <p style={{ fontWeight: 600, marginBottom: 2, color: "#f1f5f9" }}>{label}</p>
            <p style={{ color: "#38bdf8" }}>Delay rate: <strong>{payload[0].value}%</strong></p>
        </div>
    );
}

/* ── Carrier row ─────────────────────────── */
function CarrierRow({ c, rank }: { c: CarrierData; rank: number }) {
    const barColor = c.delay_rate > 22 ? "var(--rose)" : c.delay_rate > 19 ? "var(--amber)" : "var(--emerald)";
    return (
        <div style={{ display: "flex", alignItems: "center", gap: 12, padding: "9px 0", borderBottom: "1px solid rgba(255,255,255,0.03)" }}>
            <span style={{
                width: 22, height: 22, borderRadius: 6, fontSize: 11, fontWeight: 700,
                display: "flex", alignItems: "center", justifyContent: "center",
                background: rank <= 3 ? "var(--rose-muted)" : "rgba(255,255,255,0.04)",
                color: rank <= 3 ? "var(--rose)" : "var(--text-dim)",
            }}>
                {rank}
            </span>
            <div style={{ flex: 1 }}>
                <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
                    <span style={{ fontSize: 13, fontWeight: 500 }}>
                        {c.name} <span style={{ color: "var(--text-dim)" }}>({c.code})</span>
                    </span>
                    <span style={{ fontSize: 13, fontFamily: "Space Grotesk", fontWeight: 700, color: barColor }}>
                        {c.delay_rate}%
                    </span>
                </div>
                <div className="progress-track">
                    <div className="progress-fill" style={{ width: `${(c.delay_rate / 28) * 100}%`, background: barColor }} />
                </div>
            </div>
        </div>
    );
}

/* ── Dashboard page ──────────────────────── */
export default function StaffDashboard() {
    const [stats, setStats] = useState<Stats | null>(null);
    const [trends, setTrends] = useState<TrendData[]>([]);
    const [routes, setRoutes] = useState<RouteData[]>([]);
    const [carriers, setCarriers] = useState<CarrierData[]>([]);
    const [hours, setHours] = useState<HourData[]>([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        Promise.all([api.getStats(), api.getTrends(), api.getRoutes(), api.getCarriers(), api.getHours()])
            .then(([s, t, r, c, h]) => { setStats(s); setTrends(t); setRoutes(r); setCarriers(c); setHours(h); setLoading(false); })
            .catch(() => setLoading(false));
    }, []);

    if (loading) return (
        <AuthGuard>
            <Navbar />
            <div className="page" style={{ textAlign: "center", paddingTop: 120 }}>
                <span className="spin" style={{ margin: "0 auto", width: 24, height: 24, borderColor: "rgba(56,189,248,0.2)", borderTopColor: "var(--sky)" }} />
                <p style={{ color: "var(--text-secondary)", marginTop: 12, fontSize: 14 }}>Loading analytics…</p>
            </div>
        </AuthGuard>
    );

    return (
        <AuthGuard>
            <Navbar />
            <div className="page">
                {/* Header */}
                <div className="animate-enter" style={{ marginBottom: 28 }}>
                    <p style={{ fontSize: 12, fontWeight: 600, color: "var(--amber)", textTransform: "uppercase", letterSpacing: 1.2, marginBottom: 6 }}>
                        Staff Dashboard
                    </p>
                    <h1 style={{ fontSize: 28, fontWeight: 700, letterSpacing: -0.4, color: "var(--text-primary)" }}>
                        Flight Delay Analytics
                    </h1>
                </div>

                {/* Stat pills */}
                {stats && (
                    <div className="animate-enter-d1" style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 12, marginBottom: 24 }}>
                        <div className="stat-pill">
                            <div className="stat-pill-label">Delay Rate</div>
                            <div className="stat-pill-value" style={{ color: "var(--sky)" }}>{(stats.overall_delay_rate * 100).toFixed(1)}%</div>
                        </div>
                        <div className="stat-pill">
                            <div className="stat-pill-label">Airlines</div>
                            <div className="stat-pill-value">{stats.total_airlines}</div>
                        </div>
                        <div className="stat-pill">
                            <div className="stat-pill-label">Airports</div>
                            <div className="stat-pill-value">{stats.total_airports}</div>
                        </div>
                        <div className="stat-pill">
                            <div className="stat-pill-label">Routes</div>
                            <div className="stat-pill-value" style={{ color: "var(--emerald)" }}>{stats.total_routes.toLocaleString()}</div>
                        </div>
                    </div>
                )}

                {/* Row 1: hourly + DOW */}
                <div className="animate-enter-d2" style={{ display: "grid", gridTemplateColumns: "3fr 2fr", gap: 16, marginBottom: 16 }}>
                    <div className="chart-card">
                        <div className="chart-title">Delay rate by hour of day</div>
                        <ResponsiveContainer width="100%" height={240}>
                            <AreaChart data={hours}>
                                <defs>
                                    <linearGradient id="hg" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="0%" stopColor="#38bdf8" stopOpacity={0.18} />
                                        <stop offset="100%" stopColor="#38bdf8" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid stroke="rgba(255,255,255,0.03)" strokeDasharray="3 3" />
                                <XAxis dataKey="label" tick={{ fill: "#4b5563", fontSize: 10.5 }} axisLine={false} tickLine={false} interval={3} />
                                <YAxis tick={{ fill: "#4b5563", fontSize: 10.5 }} axisLine={false} tickLine={false} unit="%" width={36} />
                                <Tooltip content={<Tip />} />
                                <Area type="monotone" dataKey="delay_rate" stroke="#38bdf8" strokeWidth={2} fill="url(#hg)" dot={false} />
                            </AreaChart>
                        </ResponsiveContainer>
                    </div>

                    <div className="chart-card">
                        <div className="chart-title">Delay rate by weekday</div>
                        <ResponsiveContainer width="100%" height={240}>
                            <BarChart data={trends}>
                                <CartesianGrid stroke="rgba(255,255,255,0.03)" strokeDasharray="3 3" />
                                <XAxis dataKey="day" tick={{ fill: "#4b5563", fontSize: 10.5 }} axisLine={false} tickLine={false}
                                    tickFormatter={(v: string) => v.slice(0, 3)} />
                                <YAxis tick={{ fill: "#4b5563", fontSize: 10.5 }} axisLine={false} tickLine={false} unit="%" width={36} />
                                <Tooltip content={<Tip />} />
                                <Bar dataKey="delay_rate" radius={[4, 4, 0, 0]} barSize={28}>
                                    {trends.map((_, i) => (
                                        <Cell key={i} fill={`hsl(${200 + i * 8}, 70%, ${55 + i * 2}%)`} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>
                </div>

                {/* Row 2: routes + carriers */}
                <div className="animate-enter-d3" style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 16, marginBottom: 16 }}>
                    <div className="chart-card">
                        <div className="chart-title">Top 10 most delayed routes</div>
                        <ResponsiveContainer width="100%" height={320}>
                            <BarChart data={routes} layout="vertical" margin={{ left: 10 }}>
                                <CartesianGrid stroke="rgba(255,255,255,0.03)" strokeDasharray="3 3" />
                                <XAxis type="number" tick={{ fill: "#4b5563", fontSize: 10.5 }} axisLine={false} tickLine={false} unit="%" />
                                <YAxis dataKey="route" type="category" tick={{ fill: "#8892a4", fontSize: 10.5 }} axisLine={false} tickLine={false} width={90} />
                                <Tooltip content={<Tip />} />
                                <Bar dataKey="delay_rate" radius={[0, 4, 4, 0]} barSize={18}>
                                    {routes.map((_, i) => (
                                        <Cell key={i} fill={`hsl(${350 + i * 6}, 70%, ${60 - i * 2}%)`} />
                                    ))}
                                </Bar>
                            </BarChart>
                        </ResponsiveContainer>
                    </div>

                    <div className="chart-card" style={{ display: "flex", flexDirection: "column" }}>
                        <div className="chart-title">Airline delay ranking</div>
                        <div style={{ flex: 1, overflowY: "auto", paddingRight: 4 }}>
                            {carriers.map((c, i) => <CarrierRow key={c.code} c={c} rank={i + 1} />)}
                        </div>
                    </div>
                </div>

                {/* Insights */}
                {stats && (
                    <div className="animate-enter-d3" style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 12 }}>
                        <div className="stat-pill" style={{ borderLeft: "3px solid var(--rose)" }}>
                            <div className="stat-pill-label">Worst airline</div>
                            <div style={{ fontFamily: "Space Grotesk", fontSize: 20, fontWeight: 700, marginTop: 4 }}>{stats.worst_carrier}</div>
                            <div style={{ fontSize: 12.5, color: "var(--text-secondary)", marginTop: 2 }}>{(stats.worst_carrier_rate * 100).toFixed(1)}% delays</div>
                        </div>
                        <div className="stat-pill" style={{ borderLeft: "3px solid var(--amber)" }}>
                            <div className="stat-pill-label">Most delayed route</div>
                            <div style={{ fontFamily: "Space Grotesk", fontSize: 20, fontWeight: 700, marginTop: 4 }}>{stats.worst_route.replace("_", " → ")}</div>
                            <div style={{ fontSize: 12.5, color: "var(--text-secondary)", marginTop: 2 }}>{(stats.worst_route_rate * 100).toFixed(1)}% delays</div>
                        </div>
                        <div className="stat-pill" style={{ borderLeft: "3px solid var(--emerald)" }}>
                            <div className="stat-pill-label">Best departure hour</div>
                            <div style={{ fontFamily: "Space Grotesk", fontSize: 20, fontWeight: 700, marginTop: 4 }}>{stats.best_departure_hour}:00</div>
                            <div style={{ fontSize: 12.5, color: "var(--text-secondary)", marginTop: 2 }}>Only {(stats.best_hour_rate * 100).toFixed(1)}% delays</div>
                        </div>
                    </div>
                )}

                <p style={{ textAlign: "center", color: "var(--text-dim)", fontSize: 11.5, marginTop: 40 }}>
                    Data: Bureau of Transportation Statistics · October 2025
                </p>
            </div>
        </AuthGuard>
    );
}
