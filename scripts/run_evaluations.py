#!/usr/bin/env python3
"""
Run evaluation on PaliGemma with different configurations
"""

import subprocess
import sys
import argparse

def run_evaluation(model_name, max_samples=None, splits=None, output_prefix="baseline"):
    """Run evaluation with specified parameters."""
    
    cmd = ["python", "evaluate_baseline.py"]
    cmd.extend(["--model", model_name])
    
    if max_samples:
        cmd.extend(["--max_samples", str(max_samples)])
    
    if splits:
        cmd.extend(["--splits"] + splits)
    
    output_file = f"results/{output_prefix}_results.json"
    cmd.extend(["--output", output_file])
    
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("SUCCESS!")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("FAILED!")
        print(f"Error: {e}")
        print(f"Stdout: {e.stdout}")
        print(f"Stderr: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Run baseline evaluations")
    parser.add_argument("--quick", action="store_true", help="Quick test with limited samples")
    parser.add_argument("--full", action="store_true", help="Full evaluation on all splits")
    parser.add_argument("--split_a_only", action="store_true", help="Evaluate only Split A (factor-isolated)")
    parser.add_argument("--model", default="google/paligemma2-3b-mix-224", help="Model to evaluate")
    
    args = parser.parse_args()
    
    if args.quick:
        print("=== QUICK TEST EVALUATION ===")
        success = run_evaluation(
            model_name=args.model,
            max_samples=10,
            splits=["A"],
            output_prefix="quick_test"
        )
        
    elif args.split_a_only:
        print("=== SPLIT A ONLY EVALUATION ===")
        success = run_evaluation(
            model_name=args.model,
            splits=["A"],
            output_prefix="split_a_only"
        )
        
    elif args.full:
        print("=== FULL DATASET EVALUATION ===")
        success = run_evaluation(
            model_name=args.model,
            output_prefix="full_dataset"
        )
        
    else:
        print("Please specify --quick, --split_a_only, or --full")
        return
        
    if success:
        print("\n✓ Evaluation completed successfully!")
    else:
        print("\n✗ Evaluation failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
