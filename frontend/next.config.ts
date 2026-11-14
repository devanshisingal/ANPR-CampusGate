import type { NextConfig } from "next";

const apiUrl = (process.env.ANPR_API_URL ?? "http://127.0.0.1:8000").replace(/\/$/, "");

const nextConfig: NextConfig = {
  experimental: {
    middlewareClientMaxBodySize: "1gb",
  },
  async rewrites() {
    return [
      {
        source: "/backend/:path*",
        destination: `${apiUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
