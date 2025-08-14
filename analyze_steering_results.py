#!/usr/bin/env python3
"""
Analyze the steering vector test results to understand the effects.
"""

import json
import argparse
from collections import defaultdict, Counter

def analyze_steering_results(results_file):
    """Analyze steering vector test results."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print("=== Steering Vector Analysis ===")
    print(f"Model: {data['metadata']['model_name']}")
    print(f"Timestamp: {data['metadata']['timestamp']}")
    print(f"Samples per condition: {data['metadata']['num_samples']}")
    print(f"Strengths tested: {data['metadata']['steering_strengths']}")
    print()
    
    # Analyze patterns for each task
    for task_name, task_data in data['tasks'].items():
        print(f"\n=== TASK: {task_name.upper()} ===")
        
        # Count correct answers
        baseline_correct = 0
        total_samples = 0
        
        for layer_group in ['early', 'middle', 'late', 'all']:
            if layer_group not in task_data:
                continue
                
            print(f"\nLayer Group: {layer_group}")
            print("-" * 40)
            
            for strength_key, strength_data in task_data[layer_group].items():
                strength = strength_data['strength']
                print(f"\nSteering Strength: {strength}")
                
                correct_count = 0
                for i, sample in enumerate(strength_data['samples']):
                    baseline = sample['baseline_response']
                    steered = sample['steered_response']
                    correct = sample['correct_answer']
                    
                    # Check if steered response contains the correct answer
                    steered_correct = correct in steered.lower() if steered else False
                    baseline_correct_sample = correct in baseline.lower() if baseline else False
                    
                    if steered_correct:
                        correct_count += 1
                    
                    if i == 0:  # Count baseline accuracy from first sample of each group
                        if baseline_correct_sample:
                            baseline_correct += 1
                        total_samples += 1
                    
                    print(f"  Sample {i+1}: {sample['image_id']}")
                    print(f"    Correct: {correct}")
                    print(f"    Baseline: '{baseline[:50]}{'...' if len(baseline) > 50 else ''}'")
                    print(f"    Steered:  '{steered[:50]}{'...' if len(steered) > 50 else ''}'")
                    print(f"    Steered correct: {steered_correct}")
                
                accuracy = correct_count / len(strength_data['samples'])
                print(f"  Accuracy: {accuracy:.2f} ({correct_count}/{len(strength_data['samples'])})")
        
        print(f"\nBaseline accuracy for {task_name}: {baseline_correct/total_samples:.2f} ({baseline_correct}/{total_samples})")

def analyze_steering_patterns(results_file):
    """Analyze specific patterns in steering effects."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print("\n=== STEERING PATTERNS ===")
    
    # Analyze layer effects
    layer_effects = defaultdict(list)
    
    for task_name, task_data in data['tasks'].items():
        for layer_group, layer_data in task_data.items():
            for strength_key, strength_data in layer_data.items():
                strength = strength_data['strength']
                
                for sample in strength_data['samples']:
                    baseline = sample['baseline_response']
                    steered = sample['steered_response']
                    correct = sample['correct_answer']
                    
                    # Categorize the effect
                    if not steered:
                        effect = "empty"
                    elif correct in steered.lower():
                        effect = "correct"
                    elif len(steered) > 100:
                        effect = "repetitive"
                    elif steered == baseline:
                        effect = "no_change"
                    else:
                        effect = "changed"
                    
                    layer_effects[f"{task_name}_{layer_group}_{strength}"].append(effect)
    
    # Print pattern summary
    for condition, effects in layer_effects.items():
        effect_counts = Counter(effects)
        print(f"{condition}: {dict(effect_counts)}")

def find_best_conditions(results_file):
    """Find the best steering conditions for each task."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print("\n=== BEST CONDITIONS ===")
    
    for task_name, task_data in data['tasks'].items():
        best_accuracy = 0
        best_condition = None
        
        for layer_group, layer_data in task_data.items():
            for strength_key, strength_data in layer_data.items():
                strength = strength_data['strength']
                
                correct_count = 0
                for sample in strength_data['samples']:
                    correct = sample['correct_answer']
                    steered = sample['steered_response']
                    
                    if correct in steered.lower():
                        correct_count += 1
                
                accuracy = correct_count / len(strength_data['samples'])
                
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_condition = f"{layer_group}, strength={strength}"
        
        print(f"{task_name}: {best_condition} (accuracy: {best_accuracy:.2f})")

def main():
    parser = argparse.ArgumentParser(description='Analyze steering vector test results')
    parser.add_argument('--results', default='results/steering_test_results.json',
                        help='Path to results JSON file')
    args = parser.parse_args()
    
    analyze_steering_results(args.results)
    analyze_steering_patterns(args.results)
    find_best_conditions(args.results)

if __name__ == "__main__":
    main()
