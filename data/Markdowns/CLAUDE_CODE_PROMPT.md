# Claude Code Prompt for Training Deployment

Copy-paste this into a new Claude Code session to have it deploy your training:

---

## Prompt for New Claude Code Session

```
I have a TritonRL training implementation ready to deploy on AWS.

Working directory: /Users/Axiomatize/Documents/tritonrl
GitHub repo: https://github.com/vanshajagrawal/tritonrl

Please read IMPLEMENTATION_COMPLETE.md and LAUNCH_GUIDE.md to understand the implementation.

My constraints:
- Budget: $100 for AWS
- AWS credentials already configured
- I have vCPU limits in certain regions (check which regions are available)
- Need to deploy and run training automatically

Your tasks:
1. Find an AWS region where I have GPU capacity (check us-east-1, us-west-2, eu-west-1)
2. Launch a g5.xlarge spot instance (or g5.2xlarge if available) in that region
3. Set up the instance with all dependencies
4. Configure S3 checkpointing for spot protection
5. Start the budget-optimized training pipeline (run_budget_pipeline.sh)
6. Monitor training progress and costs
7. Auto-terminate if costs exceed $150

Key files to reference:
- LAUNCH_GUIDE.md - Complete deployment instructions
- run_budget_pipeline.sh - Budget-optimized training (200 samples, ~3 hrs)
- monitor_costs.sh - Auto-terminate at $150
- instance_userdata.sh - Auto-setup script

Expected outcome:
- Training completes in 2-3 hours
- Cost: $1-2 on spot instance
- Results show RL improves over SFT baseline by ~10%

Please proceed autonomously - don't ask for permission for AWS operations within budget.
```

---

## Alternative: Shorter Prompt

```
Deploy my TritonRL training on AWS with $100 budget.

Working dir: /Users/Axiomatize/Documents/tritonrl
Read: LAUNCH_GUIDE.md and IMPLEMENTATION_COMPLETE.md

Tasks:
1. Find AWS region with g5.xlarge capacity
2. Launch spot instance with auto-setup
3. Run training pipeline
4. Monitor costs (auto-kill at $150)

Proceed autonomously within budget.
```

---

## What Claude Code Will Do

When you give this prompt, Claude Code will:

### 1. Check AWS Regions (~2 minutes)
```bash
# Check vCPU limits across regions
aws ec2 describe-account-attributes --region us-east-1 ...
aws ec2 describe-account-attributes --region us-west-2 ...

# Check spot pricing
aws ec2 describe-spot-price-history --region us-east-1 --instance-types g5.xlarge ...
```

### 2. Launch Instance (~3 minutes)
```bash
# Get latest Deep Learning AMI
AMI_ID=$(aws ec2 describe-images --region $AWS_REGION ...)

# Launch spot instance
aws ec2 run-instances \
  --region $AWS_REGION \
  --image-id $AMI_ID \
  --instance-type g5.xlarge \
  --instance-market-options 'MarketType=spot' \
  --user-data file://instance_userdata.sh
```

### 3. Monitor Setup (~3 minutes)
```bash
# Wait for instance
aws ec2 wait instance-running ...

# Get instance details
INSTANCE_ID=$(aws ec2 describe-instances ...)
PUBLIC_IP=$(aws ec2 describe-instances ...)
```

### 4. Verify Training Started (~1 minute)
```bash
# SSH and check
ssh ubuntu@$PUBLIC_IP "tail training.log"

# Or check from outside
aws ec2 get-console-output --instance-id $INSTANCE_ID
```

### 5. Monitor Progress (2-3 hours)
Claude Code will periodically check:
```bash
# Check training progress
ssh ubuntu@$PUBLIC_IP "tail -20 training.log"

# Check costs
ssh ubuntu@$PUBLIC_IP "tail -5 cost_monitor.log"

# Check if training complete
ssh ubuntu@$PUBLIC_IP "ls -l checkpoints/rl_final/"
```

### 6. Collect Results (~5 minutes)
```bash
# Download results
scp ubuntu@$PUBLIC_IP:/home/ubuntu/tritonrl/outputs/*.json ./results/

# Show summary
cat results/eval_comparison.json
```

### 7. Terminate Instance (~1 minute)
```bash
# Terminate to stop charges
aws ec2 terminate-instances --instance-ids $INSTANCE_ID
```

---

## Expected Timeline

| Phase | Duration | Notes |
|-------|----------|-------|
| Region check | 2 min | Find available capacity |
| Instance launch | 3 min | Spot instance startup |
| Setup | 3 min | Install dependencies, clone repo |
| Training | 2-3 hrs | Full budget pipeline |
| Results collection | 5 min | Download and show results |
| Cleanup | 1 min | Terminate instance |
| **Total** | **~3 hours** | **Cost: $1-2** |

---

## What You'll Get Back

Claude Code will provide:

1. **Instance Details**
   - Region: us-east-1
   - Instance ID: i-xxxxxxxxxxxxx
   - Public IP: xx.xx.xx.xx
   - Cost rate: $0.50/hr

2. **Training Progress Updates**
   ```
   [12:00] Training started
   [12:30] Data prep complete (200 samples)
   [13:00] SFT training complete (45 min)
   [15:00] RL training complete (2 hrs)
   [15:30] Evaluation complete
   ```

3. **Final Results**
   ```json
   {
     "sft_baseline": {
       "correct": "38%",
       "valid": "85%"
     },
     "rl_model": {
       "correct": "48%",
       "valid": "90%"
     },
     "improvement": "+10%"
   }
   ```

4. **Cost Summary**
   ```
   Instance: g5.xlarge spot
   Runtime: 2.8 hours
   Rate: $0.50/hr
   Total: $1.40
   ```

---

## Troubleshooting Prompts

If something goes wrong, use these follow-up prompts:

### Issue: No Capacity
```
Try a different region or instance type.
Check: us-west-2, eu-west-1
Try: g5.2xlarge or on-demand g5.xlarge
```

### Issue: Training Fails
```
SSH into the instance and run training manually:
ssh ubuntu@$PUBLIC_IP
cd tritonrl
./run_debug_pipeline.sh  # Start with debug first
```

### Issue: Costs Too High
```
Terminate the instance immediately:
aws ec2 terminate-instances --region $REGION --instance-ids $INSTANCE_ID
```

### Issue: Need to Resume
```
The training was interrupted. Resume from S3 checkpoint:
1. Launch new instance in same region
2. Run: python resume_training.py
3. It will restore from S3 and continue
```

---

## Alternative: Manual Execution

If you prefer to run commands yourself instead of having Claude Code do it:

### Quick Manual Commands

```bash
cd /Users/Axiomatize/Documents/tritonrl

# 1. Find region with capacity
export AWS_REGION=us-east-1

# 2. Get AMI
export AMI_ID=$(aws ec2 describe-images --region $AWS_REGION --owners amazon --filters "Name=name,Values=Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04)*" --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' --output text)

# 3. Launch
aws ec2 run-instances \
  --region $AWS_REGION \
  --image-id $AMI_ID \
  --instance-type g5.xlarge \
  --instance-market-options 'MarketType=spot,SpotOptions={MaxPrice=1.00}' \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150,VolumeType=gp3}' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=tritonrl}]' \
  --user-data file://instance_userdata.sh

# 4. Get instance details
export INSTANCE_ID=$(aws ec2 describe-instances --region $AWS_REGION --filters "Name=tag:Name,Values=tritonrl" "Name=instance-state-name,Values=running,pending" --query 'Reservations[0].Instances[0].InstanceId' --output text)

export PUBLIC_IP=$(aws ec2 describe-instances --region $AWS_REGION --instance-ids $INSTANCE_ID --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)

echo "Instance: $INSTANCE_ID"
echo "IP: $PUBLIC_IP"
echo "Wait 3 minutes for setup, then check: ssh ubuntu@$PUBLIC_IP tail -f training.log"

# 5. Monitor (after 3 min)
ssh ubuntu@$PUBLIC_IP "tail -f training.log"

# 6. When done, terminate
aws ec2 terminate-instances --region $AWS_REGION --instance-ids $INSTANCE_ID
```

---

## Summary

**For autonomous deployment**: Give Claude Code the first prompt and it will handle everything.

**For manual control**: Use the quick commands above.

**Cost**: $1-2 for full proof-of-concept showing RL improves over SFT.

**Time**: 3 hours total (mostly waiting for training).

**Output**: Trained models + evaluation results proving the approach works.
