import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "CampusGate | ANPR Operations",
  description: "Live campus vehicle access monitoring",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
