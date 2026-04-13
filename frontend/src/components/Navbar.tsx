"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { useAuth } from "@/lib/auth-context";

export default function Navbar() {
    const pathname = usePathname();
    const { user, logout } = useAuth();

    const isLanding = pathname === "/";

    return (
        <header className="topbar" style={isLanding ? { background: "transparent", borderBottom: "none" } : undefined}>
            <Link href="/" className="topbar-logo">
                <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M17.8 19.2L16 11l3.5-3.5C21 6 21.5 4 21 3c-1-.5-3 0-4.5 1.5L13 8 4.8 6.2c-.5-.1-.9.1-1.1.5l-.3.5c-.2.5-.1 1 .3 1.3L9 12l-2 3H4l-1 1 3 2 2 3 1-1v-3l3-2 3.5 5.3c.3.4.8.5 1.3.3l.5-.2c.4-.3.6-.7.5-1.2z" />
                </svg>
                <span>SkyPredict</span>
            </Link>

            <nav className="topbar-nav">
                {!isLanding && (
                    <>
                        <Link href="/predict" className={`topbar-link ${pathname === "/predict" ? "active" : ""}`}>
                            Predict
                        </Link>
                        <Link href="/dashboard" className={`topbar-link ${pathname === "/dashboard" ? "active" : ""}`}>
                            Dashboard
                        </Link>
                    </>
                )}

                {user ? (
                    <div className="topbar-user">
                        {user.photoURL ? (
                            <img
                                src={user.photoURL}
                                alt={user.displayName || "User"}
                                className="topbar-avatar"
                                referrerPolicy="no-referrer"
                            />
                        ) : (
                            <div className="topbar-avatar topbar-avatar-placeholder">
                                {(user.displayName || user.email || "U").charAt(0).toUpperCase()}
                            </div>
                        )}
                        <span className="topbar-username">
                            {user.displayName || user.email || "User"}
                        </span>
                        <button
                            className="topbar-link topbar-signout"
                            onClick={logout}
                            id="signout-btn"
                        >
                            Sign Out
                        </button>
                    </div>
                ) : (
                    <Link href="/login" className={`topbar-link topbar-login-link ${isLanding ? "landing-cta-nav" : ""}`}>
                        Sign In
                    </Link>
                )}
            </nav>
        </header>
    );
}
