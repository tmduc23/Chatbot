# ---- 1. Cài thư viện (layer này được cache, chỉ chạy lại khi package*.json đổi) ----
FROM node:22-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci

# ---- 2. Test + build ----
FROM node:22-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Địa chỉ backend MÀ TRÌNH DUYỆT gọi tới (không phải địa chỉ trong mạng Docker).
# Next.js chép giá trị này vào JavaScript lúc build -> đổi thì phải build lại.
ARG NEXT_PUBLIC_API_URL=http://localhost:8000
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL \
    NEXT_TELEMETRY_DISABLED=1

# Test fail -> build dừng ở đây -> không bao giờ chạy image có lỗi
RUN npm test
# next build cũng kiểm tra TypeScript: sai kiểu -> build fail
RUN npm run build


# ---- 3. Image chạy thật ----
FROM node:22-alpine AS runner
WORKDIR /app
ENV NODE_ENV=production \
    NEXT_TELEMETRY_DISABLED=1 \
    PORT=3000 \
    HOSTNAME=0.0.0.0
COPY --from=builder --chown=node:node /app/.next/standalone ./
COPY --from=builder --chown=node:node /app/.next/static ./.next/static
# Không chạy bằng root: user "node" có sẵn trong image node
USER node
EXPOSE 3000
CMD ["node", "server.js"]