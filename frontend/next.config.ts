import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // "standalone": build ra 1 thư mục tự chạy được (server.js + đúng các package cần) -> image Docker nhỏ
  output: "standalone",
};

export default nextConfig;