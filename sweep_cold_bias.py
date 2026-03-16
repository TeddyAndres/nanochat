#!/usr/bin/env python3
"""
Sweep script for sparse-cold-bias-scale values > 4
Sequentially runs training with different bias scale values to find optimal validation BPB
"""

import subprocess
import sys
import os
from pathlib import Path

def run_training_with_bias_scale(bias_scale):
    """Run training with specified sparse-cold-bias-scale value"""
    
    cmd = [
        sys.executable, "-m", "scripts.base_train",
        "--depth=6",
        "--num-iterations=8000",
        "--device-batch-size=32",
        f"--run=sparse 65kvoc 8kstep unembedlr0.018 fasteval logit1 coldbiasclamp4_{bias_scale}",
        "--sparse-mode",
        "--window-pattern", "L",
        "--sparse-manifest", "manifests/d6_2kseq_32batch_8kstep_noaccum.json",
        "--log-every", "10",
        "--eval-every", "1000",
        "--unembedding-lr=0.018",
        "--core-metric-every", "-1",
        "--total-batch-size", "65536",
        "--sparse-logit-scale", "1",
        "--aspect-ratio", "64",
        "--warmup-ratio", "0",
        "--warmdown-ratio", "0.5",
        "--sparse-cold-bias-scale", str(bias_scale)
    ]
    
    print(f"\n{'='*60}")
    print(f"Running training with sparse-cold-bias-scale = {bias_scale}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
        print(f"✓ Completed successfully for bias_scale = {bias_scale}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed for bias_scale = {bias_scale}, error: {e}")
        return False

def main():
    """Main sweep function"""
    
    # Values to test (greater than 4)
    bias_scales = [1.8, 1.9, 2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8]
    
    print("Starting sparse-cold-bias-scale sweep")
    print(f"Testing values: {bias_scales}")
    print("Running sequentially due to VRAM constraints")
    
    results = {}
    
    for bias_scale in bias_scales:
        success = run_training_with_bias_scale(bias_scale)
        results[bias_scale] = success
        
        if not success:
            print(f"Warning: Training failed for bias_scale = {bias_scale}")
    
    print(f"\n{'='*60}")
    print("SWEEP COMPLETED")
    print(f"{'='*60}")
    print("Results summary:")
    for bias_scale, success in results.items():
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"  bias_scale = {bias_scale}: {status}")
    
    print("\nTo find the best validation BPB:")
    print("1. Check the wandb logs for each run")
    print("2. Look for the lowest validation BPB value")
    print("3. The corresponding bias_scale is the optimal value")

if __name__ == "__main__":
    main()