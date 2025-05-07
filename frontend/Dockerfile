FROM node:18-alpine as builder

WORKDIR /app

COPY package*.json ./

RUN npm install

COPY . .

ENV VITE_API_BASE_URL=https://refinsight-api.yousufaayman.com

RUN npm run build

FROM nginx:alpine

RUN apk add --no-cache gettext

COPY --from=builder /app/dist /usr/share/nginx/html

COPY nginx.conf /etc/nginx/conf.d/default.conf

# Create a script to replace environment variables in the built files
RUN echo '#!/bin/sh' > /docker-entrypoint.sh && \
    echo 'for file in $(find /usr/share/nginx/html -type f -name "*.js"); do' >> /docker-entrypoint.sh && \
    echo '  envsubst < "$file" > "$file.tmp" && mv "$file.tmp" "$file"' >> /docker-entrypoint.sh && \
    echo 'done' >> /docker-entrypoint.sh && \
    echo 'nginx -g "daemon off;"' >> /docker-entrypoint.sh && \
    chmod +x /docker-entrypoint.sh

EXPOSE 80

CMD ["/docker-entrypoint.sh"]