/** @type {import('next').NextConfig} */
const nextConfig = {
  output: "standalone",
  images: {
    remotePatterns: [
      { protocol: "http", hostname: "localhost", port: "3000", pathname: "/api/images/**" },
      { protocol: "http", hostname: "127.0.0.1", port: "3000", pathname: "/api/images/**" },
    ],
  },
};

export default nextConfig;
