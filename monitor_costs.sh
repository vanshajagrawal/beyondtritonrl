#!/bin/bash
# Monitor AWS costs and auto-terminate instances if exceeds budget

MAX_COST=150
CHECK_INTERVAL=300  # 5 minutes
INSTANCE_ID=""

echo "=========================================="
echo "AWS COST MONITOR"
echo "=========================================="
echo "Max budget: \$$MAX_COST"
echo "Check interval: ${CHECK_INTERVAL}s (5 min)"
echo "=========================================="

# Get instance ID from argument or find running instances
if [ ! -z "$1" ]; then
    INSTANCE_ID=$1
    echo "Monitoring instance: $INSTANCE_ID"
else
    echo "Finding running GPU instances..."
    INSTANCE_ID=$(aws ec2 describe-instances \
        --region ap-south-1 \
        --filters "Name=instance-state-name,Values=running" \
                  "Name=instance-type,Values=p4d.*,p5.*,g5.*" \
        --query 'Reservations[0].Instances[0].InstanceId' \
        --output text)

    if [ "$INSTANCE_ID" == "None" ] || [ -z "$INSTANCE_ID" ]; then
        echo "No running GPU instances found"
        exit 0
    fi
    echo "Found instance: $INSTANCE_ID"
fi

# Get start time
START_TIME=$(date +%s)
echo "Monitor started at: $(date)"
echo ""

while true; do
    CURRENT_TIME=$(date +%s)
    ELAPSED_HOURS=$(echo "scale=2; ($CURRENT_TIME - $START_TIME) / 3600" | bc)

    # Get instance type and spot price
    INSTANCE_INFO=$(aws ec2 describe-instances \
        --region ap-south-1 \
        --instance-ids $INSTANCE_ID \
        --query 'Reservations[0].Instances[0].[InstanceType,State.Name,SpotInstanceRequestId]' \
        --output text 2>/dev/null)

    if [ $? -ne 0 ] || [ -z "$INSTANCE_INFO" ]; then
        echo "[$(date +%H:%M:%S)] Instance $INSTANCE_ID no longer exists or accessible"
        exit 0
    fi

    INSTANCE_TYPE=$(echo $INSTANCE_INFO | awk '{print $1}')
    INSTANCE_STATE=$(echo $INSTANCE_INFO | awk '{print $2}')

    if [ "$INSTANCE_STATE" != "running" ]; then
        echo "[$(date +%H:%M:%S)] Instance $INSTANCE_ID is $INSTANCE_STATE (not running)"
        exit 0
    fi

    # Get current spot price
    SPOT_PRICE=$(aws ec2 describe-spot-price-history \
        --region ap-south-1 \
        --instance-types $INSTANCE_TYPE \
        --start-time $(date -u +%Y-%m-%dT%H:%M:%S) \
        --product-descriptions "Linux/UNIX" \
        --query 'SpotPriceHistory[0].SpotPrice' \
        --output text)

    # Calculate estimated cost
    ESTIMATED_COST=$(echo "scale=2; $SPOT_PRICE * $ELAPSED_HOURS" | bc)
    COST_PERCENT=$(echo "scale=1; $ESTIMATED_COST / $MAX_COST * 100" | bc)

    echo "[$(date +%H:%M:%S)] Cost: \$$ESTIMATED_COST / \$$MAX_COST (${COST_PERCENT}%) | Elapsed: ${ELAPSED_HOURS}h | Rate: \$${SPOT_PRICE}/hr"

    # Check if exceeded budget
    EXCEEDED=$(echo "$ESTIMATED_COST > $MAX_COST" | bc)

    if [ "$EXCEEDED" -eq 1 ]; then
        echo ""
        echo "=========================================="
        echo "⚠️  BUDGET EXCEEDED!"
        echo "=========================================="
        echo "Estimated cost: \$$ESTIMATED_COST"
        echo "Max budget: \$$MAX_COST"
        echo "Terminating instance: $INSTANCE_ID"
        echo "=========================================="

        # Terminate instance
        aws ec2 terminate-instances \
            --region ap-south-1 \
            --instance-ids $INSTANCE_ID

        echo ""
        echo "✓ Instance terminated"
        echo "Final cost estimate: \$$ESTIMATED_COST"
        echo "=========================================="

        exit 1
    fi

    # Warn at 80% budget
    WARNING_THRESHOLD=$(echo "scale=2; $MAX_COST * 0.8" | bc)
    WARNING=$(echo "$ESTIMATED_COST > $WARNING_THRESHOLD" | bc)

    if [ "$WARNING" -eq 1 ]; then
        echo "⚠️  WARNING: At ${COST_PERCENT}% of budget!"
    fi

    sleep $CHECK_INTERVAL
done
