# TritonRL Training Launch Guide

## Quick Start (Copy-Paste Ready)

### Step 1: Find a Region with GPU Capacity

```bash
# Check your vCPU limits per region (look for regions with >0 limits)
aws ec2 describe-account-attributes --region us-east-1 --attribute-names max-instances
aws ec2 describe-account-attributes --region us-west-2 --attribute-names max-instances
aws ec2 describe-account-attributes --region eu-west-1 --attribute-names max-instances

# Check spot pricing for g5.xlarge (cheapest option)
for region in us-east-1 us-west-2 eu-west-1; do
  echo "=== $region ==="
  aws ec2 describe-spot-price-history \
    --region $region \
    --instance-types g5.xlarge \
    --start-time $(date -u +%Y-%m-%dT%H:%M:%S) \
    --product-descriptions "Linux/UNIX" \
    --query 'SpotPriceHistory[0].[InstanceType,SpotPrice,AvailabilityZone]' \
    --output table
done
```

### Step 2: Launch Instance (Pick Your Region)

**Option A: Spot Instance (Cheapest - $0.30-0.50/hr)**

```bash
# Set your region
export AWS_REGION=us-east-1  # or us-west-2, eu-west-1, etc.

# Get latest Deep Learning AMI
AMI_ID=$(aws ec2 describe-images \
  --region $AWS_REGION \
  --owners amazon \
  --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" \
            "Name=state,Values=available" \
  --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
  --output text)

echo "Using AMI: $AMI_ID"

# Launch spot instance
aws ec2 run-instances \
  --region $AWS_REGION \
  --image-id $AMI_ID \
  --instance-type g5.xlarge \
  --instance-market-options 'MarketType=spot,SpotOptions={MaxPrice=1.00,SpotInstanceType=one-time}' \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=tritonrl-training}]' \
  --user-data file://$(pwd)/instance_userdata.sh
```

**Option B: On-Demand Instance (More Reliable - $1.01/hr)**

```bash
# Same as above but without instance-market-options
aws ec2 run-instances \
  --region $AWS_REGION \
  --image-id $AMI_ID \
  --instance-type g5.xlarge \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=tritonrl-training}]' \
  --user-data file://$(pwd)/instance_userdata.sh
```

### Step 3: Get Instance Details

```bash
# Get instance ID and IP
INSTANCE_ID=$(aws ec2 describe-instances \
  --region $AWS_REGION \
  --filters "Name=tag:Name,Values=tritonrl-training" "Name=instance-state-name,Values=running,pending" \
  --query 'Reservations[0].Instances[0].InstanceId' \
  --output text)

PUBLIC_IP=$(aws ec2 describe-instances \
  --region $AWS_REGION \
  --instance-ids $INSTANCE_ID \
  --query 'Reservations[0].Instances[0].PublicIpAddress' \
  --output text)

echo "Instance ID: $INSTANCE_ID"
echo "Public IP: $PUBLIC_IP"
echo ""
echo "Wait 2-3 minutes for instance to initialize..."
```

### Step 4: Monitor Training Progress

```bash
# Wait for instance to be ready
aws ec2 wait instance-running --region $AWS_REGION --instance-ids $INSTANCE_ID

# SSH into instance (wait 2-3 min after running state for setup to complete)
ssh -i ~/.ssh/your-key.pem ubuntu@$PUBLIC_IP

# Once inside:
# View training logs
tail -f training.log

# Attach to training session
tmux attach -t training

# Check cost monitor
tail -f cost_monitor.log

# Check GPU usage
watch -n 1 nvidia-smi
```

### Step 5: Manual Launch (If User-Data Failed)

If the auto-setup doesn't work, SSH in and run manually:

```bash
# SSH into instance
ssh -i ~/.ssh/your-key.pem ubuntu@$PUBLIC_IP

# Run setup
cd /home/ubuntu
git clone https://github.com/vanshajagrawal/tritonrl.git
cd tritonrl

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install triton transformers datasets accelerate peft tqdm bitsandbytes

# Setup S3 checkpointing
./setup_s3_checkpointing.sh
# Note the BUCKET_NAME output!

# Start cost monitor in background
export BUCKET_NAME=tritonrl-checkpoints-XXXXX  # Use value from above
nohup ./monitor_costs.sh $INSTANCE_ID > cost_monitor.log 2>&1 &

# Start spot monitor in background
nohup ./monitor_spot.sh $BUCKET_NAME > spot_monitor.log 2>&1 &

# Start training
tmux new-session -d -s training './run_budget_pipeline.sh 2>&1 | tee training.log'

# Attach to watch
tmux attach -t training
```

### Step 6: Cost Monitoring

The `monitor_costs.sh` script runs automatically and will:
- Check costs every 5 minutes
- Warn at 80% of $150 budget
- **Auto-terminate instance at $150** to prevent runaway costs

To manually check:

```bash
# View cost monitor log
tail -f cost_monitor.log

# Manual instance termination if needed
aws ec2 terminate-instances --region $AWS_REGION --instance-ids $INSTANCE_ID
```

---

## Budget Analysis

### g5.xlarge (1×A10G 24GB) - RECOMMENDED
- **Spot price**: $0.30-0.50/hr
- **On-demand**: $1.01/hr
- **Budget runtime**: 200 hours spot / 99 hours on-demand
- **Pipeline time**: ~8-10 hours with quantization
- **Total cost**: $3-5 spot / $10 on-demand
- **Memory**: Enough with 8-bit quantization

### g5.2xlarge (1×A10G 24GB, more CPU/RAM)
- **Spot price**: $0.57-0.59/hr
- **Total cost**: $5-6 for full pipeline

### g5.12xlarge (4×A10G 24GB) - If Available
- **Spot price**: $3.21/hr (Mumbai)
- **Pipeline time**: ~3-4 hours
- **Total cost**: $10-13

---

## Training Configurations

### Budget Configuration (Default)
**File**: `run_budget_pipeline.sh`

```bash
./run_budget_pipeline.sh
```

**Parameters**:
- 200 training samples
- 30 RL fusion tasks
- 5 samples per task
- **Time**: ~2.5-3 hours on g5.xlarge
- **Cost**: $1-2

### Full Configuration
**File**: `run_pipeline.sh`

```bash
./run_pipeline.sh
```

**Parameters**:
- 1,000 training samples
- 100 RL fusion tasks
- 10 samples per task
- **Time**: ~8-10 hours on g5.xlarge
- **Cost**: $3-5

### Debug Configuration
**File**: `run_debug_pipeline.sh`

```bash
./run_debug_pipeline.sh
```

**Parameters**:
- 20 training samples
- 5 RL fusion tasks
- 3 samples per task
- **Time**: ~30 minutes
- **Cost**: $0.25-0.50
- **Use**: Validate setup before full run

---

## Expected Results

### Budget Pipeline (200 samples)
- **SFT Baseline**: 35-40% correct on L2 tasks
- **RL Model**: 45-50% correct on L2 tasks
- **Improvement**: +10% (demonstrates RL works!)

### Full Pipeline (1,000 samples)
- **SFT Baseline**: 45% correct on L2 tasks
- **RL Model**: 55% correct on L2 tasks
- **Improvement**: +10%

---

## Troubleshooting

### Issue: Instance Launch Fails (Capacity)

**Error**: `InsufficientInstanceCapacity`

**Solution**: Try different region or smaller instance

```bash
# Try us-east-1 (usually has capacity)
export AWS_REGION=us-east-1

# Or try us-west-2
export AWS_REGION=us-west-2

# Re-run launch command
```

### Issue: vCPU Limit Exceeded

**Error**: `VcpuLimitExceeded`

**Solution**: Request limit increase or use region with approved limits

```bash
# Check your limits
aws service-quotas get-service-quota \
  --region $AWS_REGION \
  --service-code ec2 \
  --quota-code L-DB2E81BA  # Running On-Demand G instances
```

### Issue: Spot Request Exceeded

**Error**: `MaxSpotInstanceCountExceeded`

**Solution**: Use on-demand instance instead (remove spot options)

### Issue: Training OOM (Out of Memory)

**Error**: CUDA out of memory

**Solution**: Already handled by hardware-adaptive config
- Code auto-detects GPU
- Enables 8-bit quantization on g5.xlarge
- Reduces batch size automatically

### Issue: S3 Access Denied

**Error**: `NoCredentialsError` or permission denied

**Solution**:

```bash
# Check AWS credentials
aws sts get-caller-identity

# Verify S3 access
aws s3 ls

# If fails, instance needs IAM role
# Add IAM role with S3FullAccess policy to instance
```

---

## Collecting Results

### View Results

```bash
# On the instance
cat outputs/eval_comparison.json

# View checkpoints
ls -lh checkpoints/sft_final/
ls -lh checkpoints/rl_final/
```

### Download Results Locally

```bash
# From your local machine
scp -i ~/.ssh/your-key.pem ubuntu@$PUBLIC_IP:/home/ubuntu/tritonrl/outputs/*.json ./
scp -r -i ~/.ssh/your-key.pem ubuntu@$PUBLIC_IP:/home/ubuntu/tritonrl/checkpoints ./
```

### S3 Backup

Results are automatically backed up to S3 by checkpoint manager:

```bash
# List S3 backups
aws s3 ls s3://tritonrl-checkpoints-XXXXX/

# Download from S3
aws s3 sync s3://tritonrl-checkpoints-XXXXX/checkpoints ./checkpoints
```

---

## Cost Optimization Tips

1. **Use Spot Instances**: 70% cheaper than on-demand
2. **Start with Debug Pipeline**: Validate setup for $0.25 first
3. **Use g5.xlarge**: Cheapest GPU option ($0.30-0.50/hr)
4. **Monitor Costs**: `monitor_costs.sh` auto-terminates at $150
5. **Terminate When Done**: Don't forget to terminate!

```bash
# Always terminate when done
aws ec2 terminate-instances --region $AWS_REGION --instance-ids $INSTANCE_ID
```

---

## Full Command Sequence (Copy-Paste)

```bash
# 1. Set region
export AWS_REGION=us-east-1

# 2. Get AMI
AMI_ID=$(aws ec2 describe-images --region $AWS_REGION --owners amazon --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" "Name=state,Values=available" --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' --output text)

# 3. Launch instance
aws ec2 run-instances --region $AWS_REGION --image-id $AMI_ID --instance-type g5.xlarge --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3}' --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=tritonrl-training}]' --user-data file://$(pwd)/instance_userdata.sh

# 4. Get instance ID
INSTANCE_ID=$(aws ec2 describe-instances --region $AWS_REGION --filters "Name=tag:Name,Values=tritonrl-training" "Name=instance-state-name,Values=running,pending" --query 'Reservations[0].Instances[0].InstanceId' --output text)

# 5. Get IP
PUBLIC_IP=$(aws ec2 describe-instances --region $AWS_REGION --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

# 6. Wait for running
aws ec2 wait instance-running --region $AWS_REGION --instance-ids $INSTANCE_ID

# 7. SSH in (wait 2-3 min for setup)
echo "Wait 3 minutes, then SSH:"
echo "ssh -i ~/.ssh/your-key.pem ubuntu@$PUBLIC_IP"
```

---

## Summary

**Recommended path**:
1. Launch g5.xlarge in us-east-1 ($0.30-0.50/hr)
2. Run debug pipeline first (30 min, $0.25)
3. If successful, run budget pipeline (3 hrs, $1-2)
4. Total cost: **$1.25-2.50**
5. Get results showing RL improves over SFT

**Your $100 budget allows**:
- 200+ hours of g5.xlarge spot time
- Multiple full pipeline runs
- Plenty of experimentation

Good luck! 🚀
