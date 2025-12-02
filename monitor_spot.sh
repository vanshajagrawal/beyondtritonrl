#!/bin/bash
# Continuously monitor for spot interruption and sync checkpoints

BUCKET_NAME=$1

if [ -z "$BUCKET_NAME" ]; then
    echo "Usage: ./monitor_spot.sh <s3-bucket-name>"
    exit 1
fi

echo "Monitoring spot instance for interruption..."
echo "Bucket: $BUCKET_NAME"

while true; do
    # Check for spot instance termination notice
    TERMINATION=$(curl -s http://169.254.169.254/latest/meta-data/spot/instance-action 2>/dev/null)

    if [ ! -z "$TERMINATION" ]; then
        echo ""
        echo "=========================================="
        echo "⚠️  SPOT INSTANCE INTERRUPTION DETECTED!"
        echo "=========================================="
        echo "Termination notice: $TERMINATION"
        echo ""
        echo "Performing emergency checkpoint sync..."

        # Sync all checkpoints immediately
        aws s3 sync checkpoints/ s3://${BUCKET_NAME}/checkpoints/ \
            --exclude "*.log" \
            --exclude "wandb/*"

        echo ""
        echo "✓ Emergency sync complete!"
        echo "You can resume training on a new instance with:"
        echo "  export BUCKET_NAME=${BUCKET_NAME}"
        echo "  python resume_training.py"
        echo "=========================================="

        exit 0
    fi

    sleep 5  # Check every 5 seconds
done
