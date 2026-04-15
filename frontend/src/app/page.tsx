"use client";

import { useEffect, useRef, useState } from "react";
import Link from "next/link";
import Navbar from "@/components/Navbar";
import { useAuth } from "@/lib/auth-context";

/* ── Animated counter ─────────────────────── */
function Counter({ end, suffix = "", duration = 2000 }: { end: number; suffix?: string; duration?: number }) {
    const [val, setVal] = useState(0);
    const ref = useRef<HTMLDivElement>(null);

    useEffect(() => {
        const obs = new IntersectionObserver(
            ([e]) => {
                if (e.isIntersecting) {
                    const start = performance.now();
                    const step = (now: number) => {
                        const t = Math.min((now - start) / duration, 1);
                        const eased = 1 - Math.pow(1 - t, 3);
                        setVal(Math.round(eased * end));
                        if (t < 1) requestAnimationFrame(step);
                    };
                    requestAnimationFrame(step);
                    obs.disconnect();
                }
            },
            { threshold: 0.3 }
        );
        if (ref.current) obs.observe(ref.current);
        return () => obs.disconnect();
    }, [end, duration]);

    return (
        <div ref={ref} className="stat-number">
            {val.toLocaleString()}{suffix}
        </div>
    );
}

export default function LandingPage() {
    const { user } = useAuth();
    const ctaHref = user ? "/predict" : "/login";

    return (
        <>
            <Navbar />

            {/* ══════════ HERO ══════════ */}
            <section className="hero">
                <div className="hero-bg">
                    <div className="hero-glow hero-glow-1" />
                    <div className="hero-glow hero-glow-2" />
                    {/* Subtle grid lines — like a radar/ATC screen */}
                    <svg className="hero-grid-svg" viewBox="0 0 1200 600" fill="none" xmlns="http://www.w3.org/2000/svg">
                        <line x1="0" y1="150" x2="1200" y2="150" stroke="rgba(56,189,248,0.04)" />
                        <line x1="0" y1="300" x2="1200" y2="300" stroke="rgba(56,189,248,0.04)" />
                        <line x1="0" y1="450" x2="1200" y2="450" stroke="rgba(56,189,248,0.04)" />
                        <line x1="300" y1="0" x2="300" y2="600" stroke="rgba(56,189,248,0.04)" />
                        <line x1="600" y1="0" x2="600" y2="600" stroke="rgba(56,189,248,0.04)" />
                        <line x1="900" y1="0" x2="900" y2="600" stroke="rgba(56,189,248,0.04)" />
                        {/* Flight arc */}
                        <path d="M 100 420 Q 400 80 750 280 Q 950 400 1150 120" stroke="rgba(56,189,248,0.07)" strokeWidth="1.5" strokeDasharray="6 6" />
                        <circle r="3" fill="var(--sky)" opacity="0.7">
                            <animateMotion path="M 100 420 Q 400 80 750 280 Q 950 400 1150 120" dur="8s" repeatCount="indefinite" />
                        </circle>
                    </svg>
                </div>

                <div className="hero-content animate-enter">
                    <h1 className="hero-title">
                        Will your flight<br />
                        <span className="hero-title-accent">be on time?</span>
                    </h1>
                    <p className="hero-subtitle">
                        We trained an XGBoost model on U.S. domestic flight records
                        from the Bureau of Transportation Statistics — 601K+ flights
                        across 14 carriers — accurate within ±30 minutes 86.5% of the time.
                    </p>
                    <div className="hero-actions">
                        <Link href={ctaHref} className="hero-cta">
                            Check your flight
                            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M5 12h14M12 5l7 7-7 7" />
                            </svg>
                        </Link>
                        <Link href="/dashboard" className="hero-secondary">
                            Explore the data
                        </Link>
                    </div>
                </div>
            </section>

            {/* ══════════ WHAT WE BUILT ══════════ */}
            <section className="landing-section">
                <div className="built-grid">
                    {/* Left — prose explanation */}
                    <div className="built-text">
                        <p className="section-tag">The project</p>
                        <h2 className="section-title" style={{ textAlign: "left" }}>
                            A flight delay predictor,<br />backed by real data
                        </h2>
                        <p className="built-body">
                            SkyPredict isn&apos;t another dashboard template. It&apos;s a research project
                            that combines <strong>601,570 BTS flight records</strong>,&nbsp;
                            <strong>257K hourly weather observations</strong>, and a tuned
                            XGBoost model to estimate whether a given domestic flight will depart late.
                        </p>
                        <p className="built-body">
                            The model considers <strong>45 engineered features</strong> — including
                            cyclical time encodings, airport congestion proxies, carrier delay rates,
                            and holiday indicators — then outputs a delay probability and an estimated
                            delay in minutes.
                        </p>
                    </div>

                    {/* Right — stats */}
                    <div className="built-stats">
                        <div className="built-stat-card">
                            <Counter end={601} suffix="K" />
                            <span className="built-stat-label">flight records in the training set</span>
                        </div>
                        <div className="built-stat-card">
                            <Counter end={45} />
                            <span className="built-stat-label">engineered features per prediction</span>
                        </div>
                        <div className="built-stat-card">
                            <Counter end={86} suffix=".5%" />
                            <span className="built-stat-label">regression accuracy (within ±30 min)</span>
                        </div>
                        <div className="built-stat-card">
                            <Counter end={346} />
                            <span className="built-stat-label">U.S. airports covered</span>
                        </div>
                    </div>
                </div>
            </section>

            {/* ══════════ CAPABILITIES ══════════ */}
            <section className="landing-section" style={{ paddingTop: 40 }}>
                <div className="section-header">
                    <p className="section-tag">Capabilities</p>
                    <h2 className="section-title">What you can do</h2>
                </div>
                <div className="cap-grid">
                    <div className="cap-card cap-card-wide">
                        <div className="cap-icon-wrap" style={{ background: "rgba(56,189,248,0.08)" }}>
                            <svg viewBox="0 0 24 24" fill="none" stroke="var(--sky)" strokeWidth="1.5" width="28" height="28">
                                <path d="M17.8 19.2L16 11l3.5-3.5C21 6 21.5 4 21 3c-1-.5-3 0-4.5 1.5L13 8 4.8 6.2c-.5-.1-.9.1-1.1.5l-.3.5c-.2.5-.1 1 .3 1.3L9 12l-2 3H4l-1 1 3 2 2 3 1-1v-3l3-2 3.5 5.3c.3.4.8.5 1.3.3l.5-.2c.4-.3.6-.7.5-1.2z" />
                            </svg>
                        </div>
                        <h3 className="cap-title">Delay prediction</h3>
                        <p className="cap-desc">
                            Pick an airline, route, and departure time. The model classifies delay risk
                            and — if a delay is likely — estimates how many minutes to expect.
                            Uses a dual-model architecture: classifier → regressor.
                        </p>
                    </div>
                    <div className="cap-card">
                        <div className="cap-icon-wrap" style={{ background: "rgba(52,211,153,0.08)" }}>
                            <svg viewBox="0 0 24 24" fill="none" stroke="var(--emerald)" strokeWidth="1.5" width="28" height="28">
                                <path d="M3 3v18h18" /><path d="M7 16l4-8 4 4 4-6" />
                            </svg>
                        </div>
                        <h3 className="cap-title">Delay analytics</h3>
                        <p className="cap-desc">
                            See which airlines, routes, and hours have the worst delay rates.
                            Interactive charts built from the same BTS dataset.
                        </p>
                    </div>
                    <div className="cap-card">
                        <div className="cap-icon-wrap" style={{ background: "rgba(251,191,36,0.08)" }}>
                            <svg viewBox="0 0 24 24" fill="none" stroke="var(--amber)" strokeWidth="1.5" width="28" height="28">
                                <circle cx="12" cy="12" r="10" /><path d="M12 6v6l4 2" />
                            </svg>
                        </div>
                        <h3 className="cap-title">Live flight status</h3>
                        <p className="cap-desc">
                            Enter a flight number to pull real-time gate, departure, and
                            arrival data from the AviationStack API.
                        </p>
                    </div>
                </div>
            </section>

            {/* ══════════ UNDER THE HOOD ══════════ */}
            <section className="landing-section" style={{ paddingTop: 40 }}>
                <div className="section-header">
                    <p className="section-tag">Under the hood</p>
                    <h2 className="section-title">How the model works</h2>
                </div>
                <div className="hood-timeline">
                    <div className="hood-step">
                        <div className="hood-marker">1</div>
                        <div className="hood-text">
                            <h4>Feature extraction</h4>
                            <p>
                                Raw inputs (airline, airports, date, time) are transformed into 45 features:
                                cyclical sin/cos of hour and month, one-hot carrier codes, historical
                                route delay averages, airport congestion windows, holiday proximity, and more.
                            </p>
                        </div>
                    </div>
                    <div className="hood-step">
                        <div className="hood-marker">2</div>
                        <div className="hood-text">
                            <h4>Classification</h4>
                            <p>
                                An XGBoost classifier (75 Optuna trials, temporal Oct 1-25 / Oct 26-31 split)
                                predicts whether the flight will be delayed (&gt;15 min). Threshold tuned for best F1.
                            </p>
                        </div>
                    </div>
                    <div className="hood-step">
                        <div className="hood-marker">3</div>
                        <div className="hood-text">
                            <h4>Regression</h4>
                            <p>
                                If the classifier predicts delay, a second XGBoost regressor estimates
                                the expected delay duration in minutes. Both models share the same
                                feature pipeline.
                            </p>
                        </div>
                    </div>
                </div>
            </section>

            {/* ══════════ CTA ══════════ */}
            <section className="cta-section">
                <div className="cta-glow" />
                <div className="cta-content">
                    <h2 className="cta-title">Try it out</h2>
                    <p className="cta-desc">
                        Sign in with Google and run a prediction — it takes about 10 seconds.
                    </p>
                    <Link href={ctaHref} className="hero-cta" style={{ fontSize: 15, padding: "13px 28px" }}>
                        Check a flight
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M5 12h14M12 5l7 7-7 7" />
                        </svg>
                    </Link>
                </div>
            </section>

            {/* ══════════ FOOTER ══════════ */}
            <footer className="landing-footer">
                <div className="footer-inner">
                    <div className="footer-brand">
                        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="var(--sky)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M17.8 19.2L16 11l3.5-3.5C21 6 21.5 4 21 3c-1-.5-3 0-4.5 1.5L13 8 4.8 6.2c-.5-.1-.9.1-1.1.5l-.3.5c-.2.5-.1 1 .3 1.3L9 12l-2 3H4l-1 1 3 2 2 3 1-1v-3l3-2 3.5 5.3c.3.4.8.5 1.3.3l.5-.2c.4-.3.6-.7.5-1.2z" />
                        </svg>
                        <span>SkyPredict</span>
                    </div>
                    <div className="footer-links">
                        <Link href="/predict">Predict</Link>
                        <Link href="/dashboard">Analytics</Link>
                        <Link href="/login">Sign in</Link>
                    </div>
                    <div className="footer-copy">
                        Data sourced from the Bureau of Transportation Statistics · © {new Date().getFullYear()}
                    </div>
                </div>
            </footer>
        </>
    );
}
