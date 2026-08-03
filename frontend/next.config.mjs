/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  // Required by docker/frontend.Dockerfile's runtime stage.
  output: 'standalone',

  experimental: {
    typedRoutes: true,
  },

  // Proxy API calls to the FastAPI gateway in development so the browser sees a
  // same-origin URL and no CORS preflight is needed.
  async rewrites() {
    const apiBase = process.env.API_PROXY_TARGET ?? 'http://localhost:8000';
    return [
      {
        source: '/api/backend/:path*',
        destination: `${apiBase}/api/v1/:path*`,
      },
    ];
  },
};

export default nextConfig;
