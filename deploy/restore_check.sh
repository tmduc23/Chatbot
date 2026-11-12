#!/usr/bin/env bash
# Phase 13: "chưa khôi phục thử thì chưa có backup". Khôi phục 1 file backup vào database TẠM,
# đếm số dòng, rồi xoá database tạm. KHÔNG đụng vào database đang chạy.
#     ./deploy/restore_check.sh backups/db-20261020-031500.dump
set -euo pipefail
cd "$(dirname "$0")/.."
dump="${1:?Cách dùng: ./deploy/restore_check.sh backups/db-XXXX.dump}"
check_db="restore_check"

psql_in() { docker compose exec -T postgres sh -c "psql -v ON_ERROR_STOP=1 -U \"\$POSTGRES_USER\" -d $1 -tA"; }

echo "DROP DATABASE IF EXISTS $check_db; CREATE DATABASE $check_db;" | psql_in postgres
trap 'echo "DROP DATABASE IF EXISTS '"$check_db"';" | psql_in postgres >/dev/null' EXIT

docker compose exec -T postgres sh -c "pg_restore -U \"\$POSTGRES_USER\" -d $check_db --no-owner --exit-on-error" < "$dump"

echo "SELECT 'applications=' || count(*) FROM applications;
      SELECT 'documents=' || count(*) FROM documents;
      SELECT 'chunks=' || count(*) FROM chunks;
      SELECT 'chunks_without_embedding=' || count(*) FROM chunks WHERE embedding IS NULL;
      SELECT 'alembic=' || version_num FROM alembic_version;" | psql_in "$check_db"
echo "restore ok: $dump"
