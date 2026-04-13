"use client";

/**
 * SkyPredict — Authentication context provider.
 *
 * Provides auth state, Google sign-in, and logout across the app.
 */

import {
    createContext,
    useContext,
    useEffect,
    useState,
    useCallback,
    type ReactNode,
} from "react";
import {
    onAuthStateChanged,
    signInWithPopup,
    signOut,
    type User,
} from "firebase/auth";
import { auth, googleProvider } from "./firebase";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://127.0.0.1:8000";

// ─── Types ──────────────────────────────────────

interface AuthContextType {
    user: User | null;
    loading: boolean;
    error: string | null;
    signInWithGoogle: () => Promise<void>;
    logout: () => Promise<void>;
    clearError: () => void;
}

const AuthContext = createContext<AuthContextType>({
    user: null,
    loading: true,
    error: null,
    signInWithGoogle: async () => {},
    logout: async () => {},
    clearError: () => {},
});

export const useAuth = () => useContext(AuthContext);

// ─── Provider ───────────────────────────────────

export function AuthProvider({ children }: { children: ReactNode }) {
    const [user, setUser] = useState<User | null>(null);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState<string | null>(null);

    // Sync with backend after Firebase sign-in
    const syncWithBackend = useCallback(async (firebaseUser: User) => {
        try {
            const token = await firebaseUser.getIdToken();
            await fetch(`${API_BASE}/api/auth/session`, {
                method: "POST",
                headers: {
                    Authorization: `Bearer ${token}`,
                    "Content-Type": "application/json",
                },
            });
        } catch (e) {
            console.warn("[Auth] Backend session sync failed:", e);
        }
    }, []);

    // Listen for auth state changes
    useEffect(() => {
        const unsubscribe = onAuthStateChanged(auth, async (firebaseUser) => {
            setUser(firebaseUser);
            if (firebaseUser) {
                await syncWithBackend(firebaseUser);
            }
            setLoading(false);
        });
        return unsubscribe;
    }, [syncWithBackend]);

    // Google Sign-In
    const signInWithGoogle = useCallback(async () => {
        setError(null);
        try {
            await signInWithPopup(auth, googleProvider);
        } catch (e: unknown) {
            const msg = e instanceof Error ? e.message : "Google sign-in failed";
            if (msg.includes("popup-closed") || msg.includes("cancelled-popup-request")) {
                return;
            }
            setError(msg);
        }
    }, []);

    // Sign out
    const logout = useCallback(async () => {
        setError(null);
        try {
            await signOut(auth);
        } catch (e: unknown) {
            const msg = e instanceof Error ? e.message : "Sign out failed";
            setError(msg);
        }
    }, []);

    const clearError = useCallback(() => setError(null), []);

    return (
        <AuthContext.Provider value={{ user, loading, error, signInWithGoogle, logout, clearError }}>
            {children}
        </AuthContext.Provider>
    );
}
