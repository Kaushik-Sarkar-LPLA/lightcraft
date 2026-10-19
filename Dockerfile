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
RUN printf 'server {\n\
    listen 80;\n\
    server_name _;\n\
    root /usr/share/nginx/html;\n\
    index index.html;\n\
    gzip on;\n\
    gzip_static on;\n\
    gzip_types text/plain text/css application/json application/javascript text/javascript text/xml application/xml application/wasm;\n\
    location / {\n\
        try_files $uri $uri/ =404;\n\
        add_header Cross-Origin-Opener-Policy "same-origin" always;\n\
        add_header Cross-Origin-Embedder-Policy "require-corp" always;\n\
        add_header Cross-Origin-Resource-Policy "same-origin" always;\n\
        add_header Cache-Control "no-cache, no-store, must-revalidate" always;\n\
    }\n\
}\n' > /etc/nginx/conf.d/default.conf
EXPOSE 80
