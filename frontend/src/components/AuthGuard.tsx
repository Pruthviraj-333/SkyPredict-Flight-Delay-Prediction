"use client";

/**
 * SkyPredict — Route protection component.
 *
 * Wraps pages that require authentication. Redirects to /login if not signed in.
 */

import { useEffect } from "react";
import { useRouter } from "next/navigation";
import { useAuth } from "@/lib/auth-context";

export default function AuthGuard({ children }: { children: React.ReactNode }) {
    const { user, loading } = useAuth();
    const router = useRouter();

    useEffect(() => {
        if (!loading && !user) {
            router.replace("/login");
        }
    }, [user, loading, router]);

    if (loading) {
        return (
            <div
                style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    minHeight: "100vh",
                    background: "var(--bg-base)",
                    color: "var(--text-secondary)",
                    fontFamily: "Inter, sans-serif",
                    gap: 10,
                }}
            >
                <span className="spin" />
                Loading…
            </div>
        );
    }

    if (!user) {
        return null;
    }

    return <>{children}</>;
}
