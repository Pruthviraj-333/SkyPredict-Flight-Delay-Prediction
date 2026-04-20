/**
 * SkyPredict — Firebase client initialization.
 *
 * Config is loaded from environment variables prefixed with NEXT_PUBLIC_FIREBASE_.
 * Set these in frontend/.env.local (or Azure SWA configuration for production).
 *
 * Initialization is LAZY and CLIENT-ONLY to avoid errors during
 * Next.js static page prerendering (SSR).
 */

import { initializeApp, getApps, type FirebaseApp } from "firebase/app";
import { getAuth, GoogleAuthProvider, type Auth } from "firebase/auth";

const firebaseConfig = {
    apiKey: process.env.NEXT_PUBLIC_FIREBASE_API_KEY || "",
    authDomain: process.env.NEXT_PUBLIC_FIREBASE_AUTH_DOMAIN || "",
    projectId: process.env.NEXT_PUBLIC_FIREBASE_PROJECT_ID || "",
    storageBucket: process.env.NEXT_PUBLIC_FIREBASE_STORAGE_BUCKET || "",
    messagingSenderId: process.env.NEXT_PUBLIC_FIREBASE_MESSAGING_SENDER_ID || "",
    appId: process.env.NEXT_PUBLIC_FIREBASE_APP_ID || "",
};

/** Returns the Firebase app instance (client-only, null during SSR). */
export function getFirebaseApp(): FirebaseApp | null {
    if (typeof window === "undefined") return null;
    return getApps().length === 0 ? initializeApp(firebaseConfig) : getApps()[0];
}

/** Returns the Firebase Auth instance (client-only, null during SSR). */
export function getFirebaseAuth(): Auth | null {
    const app = getFirebaseApp();
    if (!app) return null;
    return getAuth(app);
}

export const googleProvider = new GoogleAuthProvider();
