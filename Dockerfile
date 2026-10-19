FROM alpine:3.20 AS fetcher
RUN apk add --no-cache curl unzip gzip
WORKDIR /app
COPY apps/lightcraft-web/prebuilt/ /tmp/prebuilt/
RUN curl -fsSL -o lightcraft-web.zip https://github.com/storytold/lightcraft/releases/download/v0.9.1/lightcraft-web-0.9.1.zip && \
    unzip lightcraft-web.zip && \
    mv lightcraft-web-0.9.1 dist && \
    rm -f dist/index.html.br dist/worker.js.br dist/lightcraft_web.js.br dist/lightcraft_web_bg.wasm.br && \
    cp -f /tmp/prebuilt/* dist/ && \
    gzip -dk -f dist/lightcraft_web_bg.wasm.gz

FROM nginx:alpine
COPY --from=fetcher /app/dist/ /usr/share/nginx/html/
RUN printf '#!/bin/sh\n\
set -eu\n\
UPSTREAM="${AGY_UPSTREAM_URL:-http://10.0.1.1:6013/v1}"\n\
AUTH_DEFAULT=""\n\
if [ -n "${AGY_API_KEY:-}" ]; then\n\
  AUTH_DEFAULT="Bearer ${AGY_API_KEY}"\n\
fi\n\
cat > /etc/nginx/conf.d/default.conf <<EOF\n\
map \\$http_authorization \\$ai_auth_header {\n\
    default \\$http_authorization;\n\
    ""      "${AUTH_DEFAULT}";\n\
}\n\
server {\n\
    listen 80;\n\
    server_name _;\n\
    root /usr/share/nginx/html;\n\
    index index.html;\n\
    client_max_body_size 10m;\n\
    gzip on;\n\
    gzip_static on;\n\
    gzip_types text/plain text/css application/json application/javascript text/javascript text/xml application/xml application/wasm;\n\
    location /api/ai/v1/ {\n\
        proxy_pass ${UPSTREAM}/;\n\
        proxy_http_version 1.1;\n\
        proxy_set_header Authorization \\$ai_auth_header;\n\
        proxy_set_header Origin https://lightroom.pixilens.app;\n\
        proxy_set_header Connection "";\n\
        proxy_buffering off;\n\
        proxy_read_timeout 600s;\n\
        proxy_send_timeout 600s;\n\
    }\n\
    location / {\n\
        try_files \\$uri \\$uri/ =404;\n\
        add_header Cross-Origin-Opener-Policy "same-origin" always;\n\
        add_header Cross-Origin-Embedder-Policy "require-corp" always;\n\
        add_header Cross-Origin-Resource-Policy "same-origin" always;\n\
        add_header Cache-Control "no-cache, no-store, must-revalidate" always;\n\
    }\n\
}\n\
EOF\n' > /docker-entrypoint.d/99-lightcraft-nginx.sh && chmod +x /docker-entrypoint.d/99-lightcraft-nginx.sh
EXPOSE 80
