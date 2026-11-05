import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "LLMarchitect — AI Model Advisor & Unsloth/LM Studio Pipeline",
  description:
    "Recommends the optimal LLM for your project by scraping Hugging Face Leaderboards & Artificial Analysis, fine-tuning with Unsloth, and serving via LM Studio.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="min-h-screen antialiased selection:bg-indigo-500 selection:text-white">
        {children}
      </body>
    </html>
  );
}
