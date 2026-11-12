#!/usr/bin/env bash
# Phase 13: backup database + file gốc đã upload. Chạy trên VPS, trong thư mục project:
#     ./deploy/backup.sh
# Cron (mỗi ngày 3:15 sáng):  15 3 * * * cd /home/deploy/rag && ./deploy/backup.sh >> backups/backup.log 2>&1
# Giữ KEEP_DAYS ngày gần nhất (mặc định 14). Backup nằm trên CÙNG VPS thì mất VPS là mất luôn:
# định kỳ chép về máy bạn (mục "Backup" trong guide).
set -euo pipefail
cd "$(dirname "$0")/.."

KEEP_DAYS="${KEEP_DAYS:-14}"
stamp="$(date +%Y%m%d-%H%M%S)"
mkdir -p backups
chmod 700 backups                 # backup chứa toàn bộ dữ liệu: chỉ chủ thư mục đọc được

# -Fc: định dạng "custom" của pg_dump (nén, khôi phục chọn lọc được bằng pg_restore)
# Biến $POSTGRES_USER/$POSTGRES_DB lấy TRONG container (dấu nháy đơn), không cần mật khẩu: pg_dump chạy cạnh database
docker compose exec -T postgres sh -c 'pg_dump -U "$POSTGRES_USER" -d "$POSTGRES_DB" -Fc' > "backups/db-$stamp.dump.part"
mv "backups/db-$stamp.dump.part" "backups/db-$stamp.dump"   # file .dump chỉ xuất hiện khi pg_dump chạy XONG

# File gốc (PDF, DOCX...) nằm trong volume storage, không nằm trong database
docker compose exec -T backend tar -C /app -czf - storage > "backups/storage-$stamp.tar.gz"

find backups -name 'db-*.dump' -mtime "+$KEEP_DAYS" -delete
find backups -name 'storage-*.tar.gz' -mtime "+$KEEP_DAYS" -delete
find backups -name '*.part' -mtime +1 -delete

echo "$(date -Is) backup ok: db-$stamp.dump ($(du -h "backups/db-$stamp.dump" | cut -f1)), storage-$stamp.tar.gz"
