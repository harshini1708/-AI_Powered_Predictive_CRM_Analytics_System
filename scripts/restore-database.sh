#!/bin/bash

# Database Restore Script
# Restores database from a backup archive

set -e

# Configuration
MONGO_HOST="${MONGO_HOST:-localhost}"
MONGO_PORT="${MONGO_PORT:-27017}"
MONGO_USER="${MONGO_USER:-ilead_admin}"
MONGO_PASSWORD="${MONGO_PASSWORD}"
MONGO_DB="${MONGO_DB:-ilead_ams}"

# Check for backup file argument
if [ -z "$1" ]; then
  echo "Usage: $0 <backup_file.archive>"
  echo "Example: $0 ./backups/backup_20260317_120000.archive"
  exit 1
fi

BACKUP_FILE="$1"

# Verify backup file exists
if [ ! -f "$BACKUP_FILE" ]; then
  echo "ERROR: Backup file not found: $BACKUP_FILE"
  exit 1
fi

echo "[$(date)] Starting database restore from: $BACKUP_FILE"
echo "[$(date)] WARNING: This will replace the existing database!"
read -p "Continue? (yes/no): " -n 3 -r
echo
if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
  echo "Restore cancelled."
  exit 1
fi

# Drop existing database
echo "[$(date)] Dropping existing database: $MONGO_DB"
if [ -z "$MONGO_PASSWORD" ]; then
  mongosh --host "$MONGO_HOST:$MONGO_PORT" --eval "db.getSiblingDB('$MONGO_DB').dropDatabase()"
else
  mongosh \
    --host "$MONGO_HOST:$MONGO_PORT" \
    --username "$MONGO_USER" \
    --password "$MONGO_PASSWORD" \
    --authenticationDatabase admin \
    --eval "db.getSiblingDB('$MONGO_DB').dropDatabase()"
fi

# Restore from backup
echo "[$(date)] Restoring database from backup..."
if [ -z "$MONGO_PASSWORD" ]; then
  mongorestore \
    --host "$MONGO_HOST:$MONGO_PORT" \
    --db "$MONGO_DB" \
    --archive="$BACKUP_FILE" \
    --gzip
else
  mongorestore \
    --host "$MONGO_HOST:$MONGO_PORT" \
    --username "$MONGO_USER" \
    --password "$MONGO_PASSWORD" \
    --authenticationDatabase admin \
    --db "$MONGO_DB" \
    --archive="$BACKUP_FILE" \
    --gzip
fi

echo "[$(date)] Database restore completed successfully!"
