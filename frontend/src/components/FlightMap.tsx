"use client";

import { useState, memo, useMemo, useCallback, useRef } from "react";
import {
    ComposableMap,
    Geographies,
    Geography,
    Marker,
    Line,
} from "react-simple-maps";

const GEO_URL = "https://cdn.jsdelivr.net/npm/us-atlas@3/states-10m.json";

export interface AirportMapItem {
    iata: string;
    name: string;
    lat: number;
    lon: number;
    delay_rate: number;
}

export interface MapRouteItem {
    origin: string;
    dest: string;
    origin_lat: number;
    origin_lon: number;
    dest_lat: number;
    dest_lon: number;
    delay_rate: number;
}

interface FlightMapProps {
    airports: AirportMapItem[];
    routes: MapRouteItem[];
}

/* ── Colour helpers ────────────────────── */
function delayColor(rate: number): string {
    if (rate < 16) return "#34d399";       // emerald
    if (rate < 20) return "#fbbf24";       // amber
    if (rate < 25) return "#fb923c";       // orange
    return "#f87171";                       // rose
}

function delayRadius(rate: number): number {
    if (rate < 16) return 3.5;
    if (rate < 20) return 4.5;
    if (rate < 25) return 5.5;
    return 6.5;
}

/* ── Tooltip component ─────────────────── */
function MapTooltip({ airport, pos, containerWidth }: { airport: AirportMapItem | null; pos: { x: number; y: number }; containerWidth: number }) {
    if (!airport) return null;
    const color = delayColor(airport.delay_rate);
    const tipWidth = 220;
    const tipHeight = 80;
    // Flip left when tooltip would overflow the container's right edge
    const flipX = pos.x + tipWidth + 20 > containerWidth;
    const flipY = pos.y > tipHeight + 20;
    return (
        <div
            style={{
                position: "absolute",
                left: flipX ? Math.max(0, pos.x - tipWidth - 14) : pos.x + 14,
                top: flipX && flipY ? pos.y - tipHeight : pos.y - 10,
                background: "#111827",
                border: `1px solid ${color}33`,
                borderRadius: 8,
                padding: "8px 12px",
                fontSize: 12,
                pointerEvents: "none",
                zIndex: 100,
                boxShadow: `0 4px 20px ${color}22`,
                maxWidth: tipWidth,
            }}
        >
            <div style={{ fontWeight: 700, color: "#f1f5f9", marginBottom: 2 }}>
                {airport.iata}
            </div>
            <div style={{ color: "#94a3b8", fontSize: 11, marginBottom: 4, lineHeight: 1.3 }}>
                {airport.name}
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <span
                    style={{
                        width: 8,
                        height: 8,
                        borderRadius: "50%",
                        background: color,
                        display: "inline-block",
                    }}
                />
                <span style={{ color, fontWeight: 600 }}>
                    {airport.delay_rate}% delay rate
                </span>
            </div>
        </div>
    );
}

/* ── Safe Marker wrapper — catches projection errors ── */
function SafeMarker({ airport, isHovered, onEnter, onLeave }: {
    airport: AirportMapItem;
    isHovered: boolean;
    onEnter: () => void;
    onLeave: () => void;
}) {
    try {
        const r = isHovered ? delayRadius(airport.delay_rate) + 2 : delayRadius(airport.delay_rate);
        return (
            <Marker
                coordinates={[airport.lon, airport.lat]}
                onMouseEnter={onEnter}
                onMouseLeave={onLeave}
            >
                <circle
                    r={r}
                    fill={delayColor(airport.delay_rate)}
                    fillOpacity={isHovered ? 1 : 0.75}
                    stroke={isHovered ? "#fff" : "none"}
                    strokeWidth={1.5}
                    style={{ cursor: "pointer", transition: "all 0.2s ease" }}
                />
                {isHovered && (
                    <text
                        textAnchor="middle"
                        y={-r - 4}
                        style={{
                            fontFamily: "Space Grotesk, sans-serif",
                            fontSize: 9,
                            fontWeight: 700,
                            fill: "#f1f5f9",
                            pointerEvents: "none",
                        }}
                    >
                        {airport.iata}
                    </text>
                )}
            </Marker>
        );
    } catch {
        return null;
    }
}

/* ── Safe Line wrapper ─────────────────── */
function SafeLine({ route, index }: { route: MapRouteItem; index: number }) {
    try {
        return (
            <Line
                key={`route-${index}`}
                from={[route.origin_lon, route.origin_lat]}
                to={[route.dest_lon, route.dest_lat]}
                stroke="#f8717188"
                strokeWidth={1.2}
                strokeLinecap="round"
            />
        );
    } catch {
        return null;
    }
}

/* ── Main map component ────────────────── */
function FlightMapInner({ airports, routes }: FlightMapProps) {
    const [hovered, setHovered] = useState<AirportMapItem | null>(null);
    const [mousePos, setMousePos] = useState({ x: 0, y: 0 });
    const containerRef = useRef<HTMLDivElement>(null);
    const [containerWidth, setContainerWidth] = useState(800);

    // Safety filter: only include airports within AlbersUsa range
    const safeAirports = useMemo(() =>
        airports.filter(a => {
            const { lat, lon } = a;
            const isCONUS = lat >= 24 && lat <= 50 && lon >= -125 && lon <= -66;
            const isAlaska = lat >= 51 && lat <= 72 && lon >= -180 && lon <= -130;
            const isHawaii = lat >= 18 && lat <= 23 && lon >= -162 && lon <= -154;
            return isCONUS || isAlaska || isHawaii;
        }),
        [airports]
    );

    const validIatas = useMemo(() => new Set(safeAirports.map(a => a.iata)), [safeAirports]);

    const safeRoutes = useMemo(() =>
        routes.filter(r => validIatas.has(r.origin) && validIatas.has(r.dest)),
        [routes, validIatas]
    );

    const handleMouseMove = useCallback((e: React.MouseEvent) => {
        if (containerRef.current) {
            const rect = containerRef.current.getBoundingClientRect();
            setMousePos({ x: e.clientX - rect.left, y: e.clientY - rect.top });
            setContainerWidth(rect.width);
        }
    }, []);

    if (safeAirports.length === 0) return null;

    return (
        <div
            ref={containerRef}
            className="chart-card animate-enter-d2"
            style={{ position: "relative", overflow: "hidden" }}
            onMouseMove={handleMouseMove}
        >
            <div className="chart-title" style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span>U.S. Airport Delay Map</span>
                <span style={{ fontSize: 11, color: "var(--text-dim)", fontWeight: 400 }}>
                    — hover for details
                </span>
            </div>

            {/* Legend */}
            <div
                style={{
                    display: "flex",
                    gap: 16,
                    fontSize: 11,
                    color: "var(--text-secondary)",
                    marginBottom: 8,
                    flexWrap: "wrap",
                }}
            >
                {[
                    { color: "#34d399", label: "< 16% (Low)" },
                    { color: "#fbbf24", label: "16–20% (Moderate)" },
                    { color: "#fb923c", label: "20–25% (Elevated)" },
                    { color: "#f87171", label: "> 25% (High)" },
                ].map((item) => (
                    <span key={item.label} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                        <span
                            style={{
                                width: 8,
                                height: 8,
                                borderRadius: "50%",
                                background: item.color,
                                display: "inline-block",
                            }}
                        />
                        {item.label}
                    </span>
                ))}
                <span style={{ display: "flex", alignItems: "center", gap: 4, marginLeft: 8 }}>
                    <span
                        style={{
                            width: 18,
                            height: 2,
                            background: "linear-gradient(90deg, #f87171, #f8717166)",
                            display: "inline-block",
                            borderRadius: 1,
                        }}
                    />
                    Top delayed routes
                </span>
            </div>

            <ComposableMap
                projection="geoAlbersUsa"
                projectionConfig={{ scale: 900 }}
                width={800}
                height={480}
                style={{ width: "100%", height: "auto" }}
            >
                {/* U.S. States */}
                <Geographies geography={GEO_URL}>
                    {({ geographies }: { geographies: any[] }) =>
                        geographies.map((geo: any) => (
                            <Geography
                                key={geo.rpiProperties?.name || geo.id || JSON.stringify(geo)}
                                geography={geo}
                                fill="#1e293b"
                                stroke="#334155"
                                strokeWidth={0.5}
                                style={{
                                    default: { outline: "none" },
                                    hover: { outline: "none", fill: "#1e3a5f" },
                                    pressed: { outline: "none" },
                                }}
                            />
                        ))
                    }
                </Geographies>

                {/* Route arcs */}
                {safeRoutes.map((route, i) => (
                    <SafeLine key={`route-${i}`} route={route} index={i} />
                ))}

                {/* Airport markers */}
                {safeAirports.map((airport) => (
                    <SafeMarker
                        key={airport.iata}
                        airport={airport}
                        isHovered={hovered?.iata === airport.iata}
                        onEnter={() => setHovered(airport)}
                        onLeave={() => setHovered(null)}
                    />
                ))}
            </ComposableMap>

            {/* Floating tooltip */}
            <MapTooltip airport={hovered} pos={mousePos} containerWidth={containerWidth} />

            {/* Stats bar */}
            <div
                style={{
                    display: "flex",
                    justifyContent: "space-between",
                    padding: "8px 0 0",
                    borderTop: "1px solid rgba(255,255,255,0.04)",
                    fontSize: 11,
                    color: "var(--text-dim)",
                }}
            >
                <span>{safeAirports.length} airports</span>
                <span>{safeRoutes.length} most delayed routes shown</span>
            </div>
        </div>
    );
}

const FlightMap = memo(FlightMapInner);
export default FlightMap;
