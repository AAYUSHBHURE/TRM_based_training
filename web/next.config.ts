import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: 'export',
  images: {
    unoptimized: true,
  },
  reactCompiler: false,
  trailingSlash: true,
};


export default nextConfig;
