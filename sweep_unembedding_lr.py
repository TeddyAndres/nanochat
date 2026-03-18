#!/usr/bin/env python3
"""
Sweep script for unembedding-lr values
Sequentially runs training with different unembedding learning rate values to find optimal validation BPB
"""

import subprocess
import sys
import os
from pathlib import Path

def run_training_with_unembedding_lr(unembedding_lr):
    """Run training with specified unembedding learning rate value"""
    
    cmd = [
        sys.executable, "-m", "scripts.base_train",
        "--depth=6",
        "--num-iterations=2000",
        "--device-batch-size=16",
        f"--run=sparse 65kvoc 2kstep 16batch 16aspect unembedlr{unembedding_lr} fasteval coldbias2.8",
        "--sparse-mode",
        "--window-pattern", "L",
        "--sparse-manifest", "manifests/d6_2kseq_16batch_aspect16_2kstep_noaccum.json",
        "--log-every", "10",
        "--eval-every", "1000",
        f"--unembedding-lr={unembedding_lr}",
        "--core-metric-every", "-1",
        "--total-batch-size", "32768",
        "--sparse-logit-scale", "1",
        "--aspect-ratio", "16",
        "--warmup-ratio", "0",
        "--warmdown-ratio", "0.5",
        "--sparse-cold-bias-scale", "2.8"
    ]
    
    print(f"\n{'='*60}")
    print(f"Running training with unembedding-lr = {unembedding_lr}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ Completed successfully for unembedding_lr = {unembedding_lr}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed for unembedding_lr = {unembedding_lr}, error: {e}")
        return False

def main():
    """Main sweep function"""
    
    # Values to test for unembedding learning rate
    unembedding_lrs = [0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.012, 0.015, 0.018, 0.02, 0.0225, 0.025, 0.0275, 0.03]
    
    print("Starting unembedding-lr sweep")
    print(f"Testing values: {unembedding_lrs}")
    print("Running sequentially due to VRAM constraints")
    
    results = {}
    
    for unembedding_lr in unembedding_lrs:
        success = run_training_with_unembedding_lr(unembedding_lr)
        results[unembedding_lr] = success
        
        if not success:
            print(f"Warning: Training failed for unembedding_lr = {unembedding_lr}")
    
    print(f"\n{'='*60}")
    print("SWEEP COMPLETED")
    print(f"{'='*60}")
    print("Results summary:")
    for unembedding_lr, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  unembedding_lr = {unembedding_lr}: {status}")
    
    print("\nTo find the best validation BPB:")
    print("1. Check the wandb logs for each run")
    print("2. Look for the lowest validation BPB value")
    print("3. The corresponding unembedding_lr is the optimal value")

if __name__ == "__main__":
    main()