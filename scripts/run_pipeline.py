#!/usr/bin/env python3
"""
Full pipeline script for visual steering experiments.

This script runs the complete pipeline:
1. Generate task vectors from dataset
2. Evaluate baseline and steered performance
3. Analyze cross-task effects
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from datetime import datetime


def run_command(cmd: list, cwd: str = None) -> tuple:
    """Run a command and return (success, stdout, stderr)."""
    # Default to project root directory
    if cwd is None:
        cwd = str(Path(__file__).parent.parent)
        
    try:
        result = subprocess.run(
            cmd, 
            cwd=cwd,
            capture_output=True, 
            text=True, 
            check=True
        )
        return True, result.stdout, result.stderr
    except subprocess.CalledProcessError as e:
        return False, e.stdout, e.stderr


def check_dataset(dataset_dir: str) -> bool:
    """Check if dataset exists and is valid."""
    dataset_path = Path(dataset_dir)
    
    required_files = [
        "manifests/images.jsonl",
        "manifests/questions.jsonl"
    ]
    
    for file_path in required_files:
        if not (dataset_path / file_path).exists():
            print(f"❌ Missing required file: {file_path}")
            return False
            
    print(f"✅ Dataset found at {dataset_dir}")
    return True


def generate_task_vectors(model_name: str, dataset_dir: str, output_dir: str,
                         max_images: int, device: str) -> bool:
    """Generate task vectors using the dataset."""
    print("\n=== Generating Task Vectors ===")
    
    cmd = [
        "python", "src/generate_task_vectors.py",
        "--model", model_name,
        "--dataset_dir", dataset_dir,
        "--output_dir", output_dir,
        "--max_images", str(max_images),
        "--device", device
    ]
    
    print(f"Running: {' '.join(cmd)}")
    success, stdout, stderr = run_command(cmd)
    
    if success:
        print("✅ Task vector generation completed successfully!")
        print(stdout)
        return True
    else:
        print("❌ Task vector generation failed!")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        return False


def evaluate_steering(model_name: str, dataset_dir: str, task_vectors_dir: str,
                     output_file: str, max_questions: int, device: str,
                     include_baseline: bool, include_cross_task: bool) -> bool:
    """Evaluate steering performance."""
    print("\n=== Evaluating Steering Performance ===")
    
    cmd = [
        "python", "src/evaluate_steering.py", 
        "--model", model_name,
        "--dataset_dir", dataset_dir,
        "--task_vectors_dir", task_vectors_dir,
        "--output_file", output_file,
        "--max_questions", str(max_questions),
        "--device", device
    ]
    
    if include_baseline:
        cmd.append("--eval_baseline")
    if include_cross_task:
        cmd.append("--eval_cross_task")
        
    print(f"Running: {' '.join(cmd)}")
    success, stdout, stderr = run_command(cmd)
    
    if success:
        print("✅ Steering evaluation completed successfully!")
        print(stdout)
        return True
    else:
        print("❌ Steering evaluation failed!")
        print(f"STDOUT: {stdout}")
        print(f"STDERR: {stderr}")
        return False


def analyze_results(results_file: str):
    """Print analysis of results."""
    print(f"\n=== Analyzing Results from {results_file} ===")
    
    if not Path(results_file).exists():
        print(f"❌ Results file not found: {results_file}")
        return
        
    try:
        with open(results_file, 'r') as f:
            results = json.load(f)
            
        print("\n📊 Summary:")
        
        # Baseline results
        if 'baseline' in results:
            print("\n🔶 Baseline Performance:")
            baseline = results['baseline']
            for split in ['A', 'B', 'C']:
                if split in baseline:
                    print(f"  Split {split}:")
                    for task in ['count', 'color', 'shape']:
                        if task in baseline[split]:
                            acc = baseline[split][task]['accuracy']
                            total = baseline[split][task]['total']
                            print(f"    {task}: {acc:.3f} ({total} questions)")
                            
        # Steering results
        if 'steering' in results:
            print("\n🎯 Steering Performance:")
            steering = results['steering']
            
            # Show best performance for each scale
            for scale in sorted(steering.keys()):
                print(f"\n  Scale {scale}:")
                for split in ['A', 'B', 'C']:
                    if split in steering[scale]:
                        print(f"    Split {split}:")
                        for task in ['count', 'color', 'shape']:
                            if task in steering[scale][split]:
                                acc = steering[scale][split][task]['accuracy']
                                total = steering[scale][split][task]['total']
                                print(f"      {task}: {acc:.3f} ({total} questions)")
                                
        # Cross-task interference
        if 'cross_task_interference' in results:
            print("\n🔄 Cross-Task Interference Summary:")
            summary = results['cross_task_interference']['summary']
            
            for task in ['count', 'color', 'shape']:
                if task in summary['on_target_effect']:
                    on_target = summary['on_target_effect'][task]['mean_accuracy']
                    cross_task = summary['cross_task_effect'][task]['mean_accuracy']
                    print(f"  {task.capitalize()} steering:")
                    print(f"    On-target effect: {on_target:.3f}")
                    print(f"    Cross-task effect: {cross_task:.3f}")
                    
    except Exception as e:
        print(f"❌ Error analyzing results: {e}")


def main():
    parser = argparse.ArgumentParser(description="Run visual steering pipeline")
    parser.add_argument("--model", type=str, default="google/paligemma2-3b-mix-224",
                       help="Model to use")
    parser.add_argument("--dataset_dir", type=str, default="data",
                       help="Dataset directory")
    parser.add_argument("--max_images", type=int, default=100,
                       help="Maximum images for vector generation")
    parser.add_argument("--max_questions", type=int, default=50,
                       help="Maximum questions per evaluation")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    parser.add_argument("--skip_vector_generation", action="store_true",
                       help="Skip task vector generation (use existing vectors)")
    parser.add_argument("--skip_evaluation", action="store_true", 
                       help="Skip evaluation (only generate vectors)")
    parser.add_argument("--no_baseline", action="store_true",
                       help="Skip baseline evaluation")
    parser.add_argument("--no_cross_task", action="store_true",
                       help="Skip cross-task interference analysis")
    parser.add_argument("--output_dir", type=str, default="experiments",
                       help="Base output directory")
    
    args = parser.parse_args()
    
    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"visual_steering_{timestamp}"
    exp_dir = Path(args.output_dir) / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🚀 Starting Visual Steering Pipeline")
    print(f"📁 Experiment directory: {exp_dir}")
    print(f"🤖 Model: {args.model}")
    print(f"📊 Dataset: {args.dataset_dir}")
    
    # Check dataset
    if not check_dataset(args.dataset_dir):
        print("❌ Dataset validation failed. Please generate dataset first:")
        print("python scripts/generate_dataset.py")
        return 1
        
    # Setup paths
    task_vectors_dir = exp_dir / "task_vectors"
    results_file = exp_dir / "results.json"
    
    # Step 1: Generate task vectors
    if not args.skip_vector_generation:
        success = generate_task_vectors(
            args.model,
            args.dataset_dir,
            str(task_vectors_dir),
            args.max_images,
            args.device
        )
        if not success:
            print("❌ Pipeline failed at vector generation step")
            return 1
    else:
        print("⏭️ Skipping task vector generation (using existing vectors)")
        if not task_vectors_dir.exists():
            print(f"❌ Task vectors directory not found: {task_vectors_dir}")
            return 1
            
    # Step 2: Evaluate steering
    if not args.skip_evaluation:
        success = evaluate_steering(
            args.model,
            args.dataset_dir,
            str(task_vectors_dir),
            str(results_file),
            args.max_questions,
            args.device,
            include_baseline=not args.no_baseline,
            include_cross_task=not args.no_cross_task
        )
        if not success:
            print("❌ Pipeline failed at evaluation step")
            return 1
    else:
        print("⏭️ Skipping evaluation")
        
    # Step 3: Analyze results
    if results_file.exists():
        analyze_results(str(results_file))
        
    # Save pipeline configuration
    config = {
        'model': args.model,
        'dataset_dir': args.dataset_dir,
        'max_images': args.max_images,
        'max_questions': args.max_questions,
        'device': args.device,
        'timestamp': timestamp,
        'experiment_dir': str(exp_dir)
    }
    
    with open(exp_dir / "config.json", 'w') as f:
        json.dump(config, f, indent=2)
        
    print(f"\n🎉 Pipeline completed successfully!")
    print(f"📁 Results saved to: {exp_dir}")
    print(f"📋 Configuration: {exp_dir / 'config.json'}")
    if results_file.exists():
        print(f"📊 Results: {results_file}")
        
    return 0


if __name__ == "__main__":
    sys.exit(main())
