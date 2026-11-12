# Postgres 17 + pgvector + pg_textsearch (BM25)
# Image gốc pgvector/pgvector:pg17 = postgres:17 chính thức + pgvector, đã cấu hình sẵn kho apt.postgresql.org
FROM pgvector/pgvector:pg17

# Ghim phiên bản để lần build sau ra đúng bản này (đổi khi muốn nâng cấp)
ARG PG_TEXTSEARCH_VERSION=v1.4.0

# Cài công cụ build -> build pg_textsearch -> xoá công cụ build cho image nhẹ
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        build-essential git ca-certificates postgresql-server-dev-17 \
 && git clone --branch "${PG_TEXTSEARCH_VERSION}" --depth 1 \
        https://github.com/timescale/pg_textsearch /tmp/pg_textsearch \
 && make -C /tmp/pg_textsearch \
 && make -C /tmp/pg_textsearch install \
 && rm -rf /tmp/pg_textsearch \
 && apt-get purge -y --auto-remove build-essential git postgresql-server-dev-17 \
 && rm -rf /var/lib/apt/lists/*