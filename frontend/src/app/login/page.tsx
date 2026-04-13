"use client";

/**
 * SkyPredict — Login page with Google Sign-In.
 */

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";

export default function LoginPage() {
    const { user, loading, error, signInWithGoogle, clearError } = useAuth();
    const router = useRouter();
    const [googleLoading, setGoogleLoading] = useState(false);

    // Redirect to home if already logged in
    useEffect(() => {
        if (!loading && user) {
            router.replace("/");
        }
    }, [user, loading, router]);

    const handleGoogle = async () => {
        setGoogleLoading(true);
        await signInWithGoogle();
        setGoogleLoading(false);
    };

    if (loading) {
        return (
            <div style={{
                display: "flex", alignItems: "center", justifyContent: "center",
                minHeight: "100vh", background: "var(--bg-base)", color: "var(--text-secondary)",
                fontFamily: "Inter, sans-serif", gap: 10,
            }}>
                <span className="spin" />Loading…
            </div>
        );
    }

    if (user) return null;

    return (
        <div className="login-page">
            {/* Animated background orbs */}
            <div className="login-bg">
                <div className="login-orb login-orb-1" />
                <div className="login-orb login-orb-2" />
                <div className="login-orb login-orb-3" />
            </div>

            <div className="login-card animate-enter">
                {/* Logo */}
                <div className="login-logo">
                    <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="var(--sky)" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                        <path d="M17.8 19.2L16 11l3.5-3.5C21 6 21.5 4 21 3c-1-.5-3 0-4.5 1.5L13 8 4.8 6.2c-.5-.1-.9.1-1.1.5l-.3.5c-.2.5-.1 1 .3 1.3L9 12l-2 3H4l-1 1 3 2 2 3 1-1v-3l3-2 3.5 5.3c.3.4.8.5 1.3.3l.5-.2c.4-.3.6-.7.5-1.2z" />
                    </svg>
                    <h1 className="login-title">SkyPredict</h1>
                    <p className="login-subtitle">Sign in to predict flight delays</p>
                </div>

                {/* Error message */}
                {error && (
                    <div className="login-error" onClick={clearError}>
                        <span>{error}</span>
                        <span style={{ opacity: 0.5, fontSize: 11 }}>click to dismiss</span>
                    </div>
                )}

                {/* Google Sign-In */}
                <button
                    className="login-btn login-btn-google"
                    onClick={handleGoogle}
                    disabled={googleLoading}
                    id="google-signin-btn"
                >
                    {googleLoading ? (
                        <span className="spin" />
                    ) : (
                        <svg width="20" height="20" viewBox="0 0 24 24">
                            <path d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 01-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1z" fill="#4285F4" />
                            <path d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.98.66-2.23 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.84C3.99 20.53 7.7 23 12 23z" fill="#34A853" />
                            <path d="M5.84 14.09a7.2 7.2 0 010-4.18V7.07H2.18A11.97 11.97 0 001 12c0 1.94.46 3.77 1.18 5.07l3.66-2.84z" fill="#FBBC05" />
                            <path d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1 7.7 1 3.99 3.47 2.18 7.07l3.66 2.84c.87-2.6 3.3-4.53 6.16-4.53z" fill="#EA4335" />
                        </svg>
                    )}
                    <span>Continue with Google</span>
                </button>

                {/* Features teaser */}
                <div style={{ marginTop: 28, display: "flex", flexDirection: "column", gap: 10 }}>
                    {[
                        { icon: "✈️", text: "Predict delays for 600K+ flight routes" },
                        { icon: "📊", text: "Real-time analytics dashboard" },
                        { icon: "🔍", text: "Live flight tracking" },
                    ].map((f, i) => (
                        <div key={i} style={{
                            display: "flex", alignItems: "center", gap: 10,
                            fontSize: 12.5, color: "var(--text-secondary)",
                        }}>
                            <span style={{ fontSize: 14 }}>{f.icon}</span>
                            <span>{f.text}</span>
                        </div>
                    ))}
                </div>

                {/* Footer */}
                <p className="login-footer">
                    Secure sign-in powered by Google
                </p>
            </div>
        </div>
    );
}
