import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "SkyPredict — Flight Delay Prediction",
  description: "AI-powered flight delay prediction using machine learning. Predict if your flight will be delayed before you fly.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
