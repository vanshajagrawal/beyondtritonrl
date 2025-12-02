#!/bin/bash
# Complete setup script for S3 checkpointing

set -e

echo "=========================================="
echo "S3 CHECKPOINT SETUP"
echo "=========================================="

# 1. Create S3 bucket
export BUCKET_NAME="tritonrl-checkpoints-$(date +%s)"
echo "Creating S3 bucket: ${BUCKET_NAME}"

aws s3 mb s3://${BUCKET_NAME}

# 2. Enable versioning
echo "Enabling versioning..."
aws s3api put-bucket-versioning \
    --bucket ${BUCKET_NAME} \
    --versioning-configuration Status=Enabled

# 3. Enable encryption
echo "Enabling encryption..."
aws s3api put-bucket-encryption \
    --bucket ${BUCKET_NAME} \
    --server-side-encryption-configuration '{
        "Rules": [{
            "ApplyServerSideEncryptionByDefault": {
                "SSEAlgorithm": "AES256"
            }
        }]
    }'

# 4. Block public access
echo "Blocking public access..."
aws s3api put-public-access-block \
    --bucket ${BUCKET_NAME} \
    --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true"

# 5. Set lifecycle policy (delete old checkpoints after 7 days)
echo "Setting lifecycle policy..."
aws s3api put-bucket-lifecycle-configuration \
    --bucket ${BUCKET_NAME} \
    --lifecycle-configuration '{
        "Rules": [{
            "Id": "DeleteOldCheckpoints",
            "Status": "Enabled",
            "Prefix": "checkpoints/",
            "Expiration": {"Days": 7}
        }]
    }'

# 6. Save bucket name
echo "export BUCKET_NAME=${BUCKET_NAME}" >> ~/.bashrc
echo "export BUCKET_NAME=${BUCKET_NAME}" >> ~/.bash_profile

# 7. Make scripts executable
chmod +x checkpoint_manager.py monitor_spot.sh resume_training.py

echo ""
echo "=========================================="
echo "✓ SETUP COMPLETE"
echo "=========================================="
echo ""
echo "Bucket created: ${BUCKET_NAME}"
echo ""
echo "To use in current session:"
echo "  export BUCKET_NAME=${BUCKET_NAME}"
echo ""
echo "Start training with checkpointing:"
echo "  # 1. Start spot monitor (in background)"
echo "  nohup ./monitor_spot.sh ${BUCKET_NAME} > spot_monitor.log 2>&1 &"
echo ""
echo "  # 2. Run training"
echo "  python train_integrated.py --stage all"
echo ""
echo "If interrupted, resume on new instance:"
echo "  export BUCKET_NAME=${BUCKET_NAME}"
echo "  python resume_training.py"
echo ""
echo "=========================================="

# Export for current session
export BUCKET_NAME=${BUCKET_NAME}
