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
                        const eased = 1 - Math.pow(1 - t, 3); // easeOutCubic
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

/* ── Feature card icons ───────────────────── */
const features = [
    {
        icon: (
            <svg viewBox="0 0 24 24" fill="none" stroke="var(--sky)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" width="32" height="32">
                <path d="M12 2v4m0 12v4M4.93 4.93l2.83 2.83m8.48 8.48l2.83 2.83M2 12h4m12 0h4M4.93 19.07l2.83-2.83m8.48-8.48l2.83-2.83" />
            </svg>
        ),
        title: "AI-Powered Prediction",
        desc: "XGBoost models trained on 600K+ flights with 90%+ accuracy predict delays before they happen.",
        color: "var(--sky)",
    },
    {
        icon: (
            <svg viewBox="0 0 24 24" fill="none" stroke="var(--emerald)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" width="32" height="32">
                <rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" />
                <rect x="3" y="14" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" />
            </svg>
        ),
        title: "Analytics Dashboard",
        desc: "Explore delay patterns by airline, route, hour, and day of week through interactive visualizations.",
        color: "var(--emerald)",
    },
    {
        icon: (
            <svg viewBox="0 0 24 24" fill="none" stroke="var(--amber)" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" width="32" height="32">
                <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
            </svg>
        ),
        title: "Live Flight Tracking",
        desc: "Real-time flight status with gate info, departure times, and delay alerts via AviationStack API.",
        color: "var(--amber)",
    },
];

/* ── Steps ────────────────────────────────── */
const steps = [
    { num: "01", title: "Enter Flight Details", desc: "Select your airline, origin, destination, and departure time." },
    { num: "02", title: "AI Analyzes Patterns", desc: "Our ML model evaluates 45+ features from historical flight data." },
    { num: "03", title: "Get Instant Prediction", desc: "Receive your delay probability with risk assessment and insights." },
];

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
                    <div className="hero-glow hero-glow-3" />
                    {/* Animated flight path */}
                    <svg className="hero-flight-path" viewBox="0 0 800 400" fill="none">
                        <path
                            d="M -50 350 Q 200 50 400 200 Q 600 350 850 50"
                            stroke="rgba(56,189,248,0.08)"
                            strokeWidth="2"
                            strokeDasharray="8 8"
                        />
                        <circle r="4" fill="var(--sky)" opacity="0.8">
                            <animateMotion
                                path="M -50 350 Q 200 50 400 200 Q 600 350 850 50"
                                dur="6s"
                                repeatCount="indefinite"
                            />
                        </circle>
                    </svg>
                </div>

                <div className="hero-content animate-enter">
                    <div className="hero-badge">
                        <span className="hero-badge-dot" />
                        ML-Powered Flight Intelligence
                    </div>
                    <h1 className="hero-title">
                        Predict Flight Delays<br />
                        <span className="hero-title-accent">Before They Happen</span>
                    </h1>
                    <p className="hero-subtitle">
                        SkyPredict uses machine learning trained on 600,000+ flights to give you
                        instant delay predictions with 90%+ accuracy.
                    </p>
                    <div className="hero-actions">
                        <Link href={ctaHref} className="hero-cta">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                                <path d="M17.8 19.2L16 11l3.5-3.5C21 6 21.5 4 21 3c-1-.5-3 0-4.5 1.5L13 8 4.8 6.2c-.5-.1-.9.1-1.1.5l-.3.5c-.2.5-.1 1 .3 1.3L9 12l-2 3H4l-1 1 3 2 2 3 1-1v-3l3-2 3.5 5.3c.3.4.8.5 1.3.3l.5-.2c.4-.3.6-.7.5-1.2z" />
                            </svg>
                            Get Started Free
                        </Link>
                        <Link href="/dashboard" className="hero-secondary">
                            View Analytics →
                        </Link>
                    </div>
                </div>
            </section>

            {/* ══════════ STATS BAR ══════════ */}
            <section className="stats-bar animate-enter-d1">
                <div className="stats-bar-inner">
                    {[
                        { value: 600, suffix: "K+", label: "Flights Analyzed" },
                        { value: 92, suffix: "%", label: "Prediction Accuracy" },
                        { value: 20, suffix: "+", label: "Airlines Covered" },
                        { value: 340, suffix: "+", label: "Airports" },
                    ].map((s, i) => (
                        <div key={i} className="stats-bar-item">
                            <Counter end={s.value} suffix={s.suffix} />
                            <div className="stats-bar-label">{s.label}</div>
                        </div>
                    ))}
                </div>
            </section>

            {/* ══════════ FEATURES ══════════ */}
            <section className="landing-section">
                <div className="section-header animate-enter-d2">
                    <p className="section-tag">Features</p>
                    <h2 className="section-title">Everything You Need to Stay Ahead</h2>
                    <p className="section-desc">
                        From prediction to analytics, SkyPredict gives you the complete flight intelligence toolkit.
                    </p>
                </div>
                <div className="features-grid animate-enter-d3">
                    {features.map((f, i) => (
                        <div key={i} className="feature-card">
                            <div className="feature-icon" style={{ background: `${f.color}15` }}>
                                {f.icon}
                            </div>
                            <h3 className="feature-title">{f.title}</h3>
                            <p className="feature-desc">{f.desc}</p>
                        </div>
                    ))}
                </div>
            </section>

            {/* ══════════ HOW IT WORKS ══════════ */}
            <section className="landing-section">
                <div className="section-header">
                    <p className="section-tag">How It Works</p>
                    <h2 className="section-title">Three Steps to Smarter Travel</h2>
                </div>
                <div className="steps-grid">
                    {steps.map((s, i) => (
                        <div key={i} className="step-card">
                            <div className="step-num">{s.num}</div>
                            <h3 className="step-title">{s.title}</h3>
                            <p className="step-desc">{s.desc}</p>
                            {i < steps.length - 1 && <div className="step-connector" />}
                        </div>
                    ))}
                </div>
            </section>

            {/* ══════════ CTA ══════════ */}
            <section className="cta-section">
                <div className="cta-glow" />
                <div className="cta-content">
                    <h2 className="cta-title">Ready to Predict Your Next Flight?</h2>
                    <p className="cta-desc">
                        Join SkyPredict and get instant delay predictions powered by machine learning.
                        Free to use, no credit card required.
                    </p>
                    <Link href={ctaHref} className="hero-cta" style={{ fontSize: 16, padding: "14px 32px" }}>
                        Start Predicting Now
                    </Link>
                    <div className="cta-trust">
                        <span>🔒 Secure Google Sign-In</span>
                        <span>⚡ Instant Results</span>
                        <span>🎓 Free for Students</span>
                    </div>
                </div>
            </section>

            {/* ══════════ FOOTER ══════════ */}
            <footer className="landing-footer">
                <div className="footer-inner">
                    <div className="footer-brand">
                        <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--sky)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                            <path d="M17.8 19.2L16 11l3.5-3.5C21 6 21.5 4 21 3c-1-.5-3 0-4.5 1.5L13 8 4.8 6.2c-.5-.1-.9.1-1.1.5l-.3.5c-.2.5-.1 1 .3 1.3L9 12l-2 3H4l-1 1 3 2 2 3 1-1v-3l3-2 3.5 5.3c.3.4.8.5 1.3.3l.5-.2c.4-.3.6-.7.5-1.2z" />
                        </svg>
                        <span>SkyPredict</span>
                    </div>
                    <div className="footer-links">
                        <Link href="/predict">Predict</Link>
                        <Link href="/dashboard">Dashboard</Link>
                        <Link href="/login">Sign In</Link>
                    </div>
                    <div className="footer-copy">
                        © {new Date().getFullYear()} SkyPredict · Built with ML & ❤️ · Data: Bureau of Transportation Statistics
                    </div>
                </div>
            </footer>
        </>
    );
}
