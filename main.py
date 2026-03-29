#!/usr/bin/env python3
"""
main.py — Orchestrator that:
  1. Starts a RunPod pod
  2. Uploads the MNIST training code
  3. Executes training
  4. Downloads outputs (model + metrics)
  5. Terminates the pod

Usage:
    export RUNPOD_API_KEY="your_key_here"
    python main.py

    # With custom GPU for production (H100):
    python main.py --gpu "NVIDIA H100 80GB HBM3"

    # On-demand instead of spot:
    python main.py --on-demand
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from runpod_helpers import run_training, DEFAULT_GPU_TYPE


def main():
    parser = argparse.ArgumentParser(description="Run MNIST training on RunPod GPU")
    parser.add_argument(
        "--code-dir",
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_code"),
        help="Path to training code directory",
    )
    parser.add_argument("--script", default="train_mnist.py", help="Training script filename")
    parser.add_argument("--output-dir", default="./outputs", help="Local directory for outputs")
    parser.add_argument("--gpu", default=DEFAULT_GPU_TYPE, help="GPU type ID")
    parser.add_argument("--on-demand", action="store_true", help="Use on-demand (not spot)")
    parser.add_argument("--name", default="mnist-training", help="Pod name")

    args = parser.parse_args()

    api_key = os.environ.get("RUNPOD_API_KEY")
    if not api_key:
        print("ERROR: Set RUNPOD_API_KEY environment variable")
        sys.exit(1)

    print("=" * 60)
    print("RunPod MNIST Training Orchestrator")
    print("=" * 60)
    print(f"  GPU:       {args.gpu}")
    print(f"  Spot:      {not args.on_demand}")
    print(f"  Code dir:  {args.code_dir}")
    print(f"  Script:    {args.script}")
    print(f"  Output:    {args.output_dir}")
    print("=" * 60)

    result = run_training(
        training_code_dir=args.code_dir,
        train_script=args.script,
        output_dir=args.output_dir,
        gpu_type=args.gpu,
        spot=not args.on_demand,
        pod_name=args.name,
    )

    print("\n" + "=" * 60)
    if result["success"]:
        print("TRAINING COMPLETED SUCCESSFULLY")
        m = result.get("metrics", {})
        if m:
            print(f"  Final test accuracy: {m.get('final_test_accuracy', '?')}%")
            print(f"  Training time:       {m.get('training_time_seconds', '?')}s")
            print(f"  GPU used:            {m.get('gpu_name', '?')}")
        print(f"  Outputs saved to:    {result['output_dir']}")
    else:
        print("TRAINING FAILED")
        print(f"  Error: {result.get('error', 'Unknown')}")
    print("=" * 60)

    return 0 if result["success"] else 1


if __name__ == "__main__":
    sys.exit(main())
