#!/bin/bash

# Database Backup Script
# Creates a timestamped backup of the MongoDB database

set -e

# Configuration
BACKUP_DIR="${BACKUP_DIR:-.}/backups"
MONGO_HOST="${MONGO_HOST:-localhost}"
MONGO_PORT="${MONGO_PORT:-27017}"
MONGO_USER="${MONGO_USER:-ilead_admin}"
MONGO_PASSWORD="${MONGO_PASSWORD}"
MONGO_DB="${MONGO_DB:-ilead_ams}"
BACKUP_RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Generate timestamp
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/backup_${TIMESTAMP}.archive"
LOG_FILE="$BACKUP_DIR/backup_${TIMESTAMP}.log"

echo "[$(date)] Starting database backup..." | tee "$LOG_FILE"

# Create backup
if [ -z "$MONGO_PASSWORD" ]; then
  mongodump \
    --host "$MONGO_HOST:$MONGO_PORT" \
    --db "$MONGO_DB" \
    --archive="$BACKUP_FILE" \
    --gzip 2>&1 | tee -a "$LOG_FILE"
else
  mongodump \
    --host "$MONGO_HOST:$MONGO_PORT" \
    --username "$MONGO_USER" \
    --password "$MONGO_PASSWORD" \
    --authenticationDatabase admin \
    --db "$MONGO_DB" \
    --archive="$BACKUP_FILE" \
    --gzip 2>&1 | tee -a "$LOG_FILE"
fi

# Check if backup was successful
if [ -f "$BACKUP_FILE" ]; then
  SIZE=$(du -h "$BACKUP_FILE" | cut -f1)
  echo "[$(date)] Backup completed successfully. Size: $SIZE" | tee -a "$LOG_FILE"
  
  # Upload to S3 if configured
  if command -v aws &> /dev/null && [ ! -z "$AWS_S3_BUCKET" ]; then
    echo "[$(date)] Uploading backup to S3..." | tee -a "$LOG_FILE"
    aws s3 cp "$BACKUP_FILE" "s3://${AWS_S3_BUCKET}/backups/" --region "${AWS_REGION:-us-east-1}" 2>&1 | tee -a "$LOG_FILE"
  fi
else
  echo "[$(date)] ERROR: Backup failed!" | tee -a "$LOG_FILE"
  exit 1
fi

# Cleanup old backups
echo "[$(date)] Cleaning up backups older than $BACKUP_RETENTION_DAYS days..." | tee -a "$LOG_FILE"
find "$BACKUP_DIR" -name "backup_*.archive" -mtime +$BACKUP_RETENTION_DAYS -delete

echo "[$(date)] Backup process completed." | tee -a "$LOG_FILE"
