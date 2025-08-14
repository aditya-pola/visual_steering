"""
Evaluate model performance with task vector steering.

This script tests how steering vectors affect model predictions on our visual dataset,
comparing baseline vs steered performance across different splits.
"""

import json
import torch
import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from PIL import Image
import numpy as np
from tqdm import tqdm

from .paligemma_wrapper import PaliGemmaWrapper


class SteeringEvaluator:
    """
    Evaluator for steering-based model performance on visual tasks.
    """
    
    def __init__(self, model_name: str, dataset_dir: str, task_vectors_dir: str, 
                 device: str = "auto"):
        self.model_name = model_name
        self.dataset_dir = Path(dataset_dir)
        self.task_vectors_dir = Path(task_vectors_dir)
        self.device = device
        
        # Load MLLM wrapper
        print(f"Initializing {model_name}...")
        self.model = PaliGemmaWrapper(model_name, device)
        
        # Load task vectors
        self.task_vectors = self.load_task_vectors()
        
        # Load dataset
        self.images = {}
        self.questions = []
        self.load_dataset()
        
    def load_task_vectors(self) -> Dict[str, Dict[str, torch.Tensor]]:
        """Load pre-generated task vectors."""
        print("Loading task vectors...")
        
        task_vectors = {}
        metadata_path = self.task_vectors_dir / "metadata.json"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Task vector metadata not found: {metadata_path}")
            
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            
        for task in metadata['tasks']:
            task_vectors[task] = {}
            task_dir = self.task_vectors_dir / task
            
            if not task_dir.exists():
                print(f"Warning: Task directory {task_dir} not found")
                continue
                
            # Load all vector files for this task
            for vector_file in task_dir.glob("vector_*.pt"):
                # Extract layer name from filename
                layer_name = vector_file.stem.replace("vector_", "").replace("_", ".")
                
                # Handle special cases in layer name conversion
                layer_name = layer_name.replace(".language_model.model.layers", "language_model.model.layers")
                
                vector = torch.load(vector_file, map_location=self.device)
                task_vectors[task][layer_name] = vector
                
            print(f"Loaded {len(task_vectors[task])} vectors for task '{task}'")
            
        return task_vectors
        
    def load_dataset(self):
        """Load the visual dataset."""
        images_file = self.dataset_dir / "manifests" / "images.jsonl"
        questions_file = self.dataset_dir / "manifests" / "questions.jsonl"
        
        print("Loading dataset...")
        
        # Load image metadata
        with open(images_file, 'r') as f:
            for line in f:
                img_data = json.loads(line)
                self.images[img_data['image_id']] = img_data
                
        # Load questions
        with open(questions_file, 'r') as f:
            for line in f:
                question_data = json.loads(line)
                self.questions.append(question_data)
                
        print(f"Loaded {len(self.images)} images and {len(self.questions)} questions")
        
    def load_image(self, image_id: str) -> Image.Image:
        """Load an image by ID."""
        if image_id not in self.images:
            raise ValueError(f"Image {image_id} not found")
            
        img_info = self.images[image_id]
        img_path = self.dataset_dir / img_info['render']['file']
        
        return Image.open(img_path).convert('RGB')
        
    def get_questions_by_split_and_task(self, split: str, task: str) -> List[Dict]:
        """Get questions for a specific split and task."""
        return [q for q in self.questions if q['split'] == split and q['task'] == task]
        
    def evaluate_baseline(self, questions: List[Dict], max_questions: Optional[int] = None) -> Dict:
        """Evaluate baseline model performance without steering."""
        if max_questions is not None:
            questions = questions[:max_questions]
            
        correct = 0
        total = len(questions)
        predictions = []
        
        print(f"Evaluating baseline on {total} questions...")
        
        for question in tqdm(questions, desc="Baseline evaluation"):
            try:
                image = self.load_image(question['image_id'])
                prompt = question['prompt_text']
                
                # Generate answer without steering
                response = self.model.generate(image, prompt, max_new_tokens=10)
                
                # Extract answer (model tends to repeat prompt, take last word)
                pred_answer = response.strip().split()[-1].lower()
                true_answer = question['answer'].lower()
                
                is_correct = pred_answer == true_answer
                if is_correct:
                    correct += 1
                    
                predictions.append({
                    'question_id': question.get('question_id', ''),
                    'image_id': question['image_id'],
                    'prompt': prompt,
                    'prediction': pred_answer,
                    'answer': true_answer,
                    'correct': is_correct
                })
                
            except Exception as e:
                print(f"Error processing question {question.get('question_id', '')}: {e}")
                predictions.append({
                    'question_id': question.get('question_id', ''),
                    'image_id': question['image_id'], 
                    'prompt': question['prompt_text'],
                    'prediction': 'ERROR',
                    'answer': question['answer'],
                    'correct': False
                })
                
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'predictions': predictions
        }
        
    def evaluate_with_steering(self, questions: List[Dict], task: str, 
                             steering_scale: float = 1.0,
                             max_questions: Optional[int] = None) -> Dict:
        """Evaluate model performance with task vector steering."""
        if task not in self.task_vectors:
            raise ValueError(f"Task vectors not available for task: {task}")
            
        if max_questions is not None:
            questions = questions[:max_questions]
            
        correct = 0
        total = len(questions)
        predictions = []
        
        print(f"Evaluating with {task} steering (scale={steering_scale}) on {total} questions...")
        
        for question in tqdm(questions, desc=f"{task.capitalize()} steering"):
            try:
                image = self.load_image(question['image_id'])
                prompt = question['prompt_text']
                
                # Generate answer with steering
                response = self.model.generate_with_steering(
                    image=image,
                    prompt=prompt,
                    steering_vectors=self.task_vectors[task],
                    steering_scale=steering_scale,
                    max_new_tokens=10
                )
                
                # Extract answer
                pred_answer = response.strip().split()[-1].lower()
                true_answer = question['answer'].lower()
                
                is_correct = pred_answer == true_answer
                if is_correct:
                    correct += 1
                    
                predictions.append({
                    'question_id': question.get('question_id', ''),
                    'image_id': question['image_id'],
                    'prompt': prompt,
                    'prediction': pred_answer,
                    'answer': true_answer,
                    'correct': is_correct
                })
                
            except Exception as e:
                print(f"Error processing question {question.get('question_id', '')}: {e}")
                predictions.append({
                    'question_id': question.get('question_id', ''),
                    'image_id': question['image_id'],
                    'prompt': question['prompt_text'], 
                    'prediction': 'ERROR',
                    'answer': question['answer'],
                    'correct': False
                })
                
        accuracy = correct / total if total > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total,
            'predictions': predictions,
            'steering_task': task,
            'steering_scale': steering_scale
        }
        
    def evaluate_cross_task_interference(self, max_questions_per_task: int = 50,
                                       steering_scales: List[float] = [0.5, 1.0, 2.0]) -> Dict:
        """
        Test cross-task interference: how does steering for one task affect other tasks?
        
        Args:
            max_questions_per_task: Max questions to test per task
            steering_scales: Different steering strengths to test
            
        Returns:
            Results dictionary with cross-task interference analysis
        """
        print("=== Cross-Task Interference Analysis ===")
        
        tasks = ['count', 'color', 'shape']
        splits_to_test = ['B', 'C']  # Mixed and stress splits
        
        results = {
            'splits': {},
            'summary': {}
        }
        
        for split in splits_to_test:
            print(f"\nTesting split {split}")
            results['splits'][split] = {}
            
            for target_task in tasks:
                if target_task not in self.task_vectors:
                    continue
                    
                # Get questions for each task in this split
                task_questions = {}
                for task in tasks:
                    questions = self.get_questions_by_split_and_task(split, task)
                    if len(questions) > max_questions_per_task:
                        questions = questions[:max_questions_per_task]
                    task_questions[task] = questions
                    
                results['splits'][split][target_task] = {}
                
                # Test different steering scales
                for scale in steering_scales:
                    print(f"  Steering: {target_task} (scale={scale})")
                    scale_results = {}
                    
                    # Test effect on each task
                    for eval_task in tasks:
                        if len(task_questions[eval_task]) == 0:
                            continue
                            
                        # Evaluate with steering
                        eval_result = self.evaluate_with_steering(
                            task_questions[eval_task], 
                            target_task, 
                            scale,
                            max_questions=None
                        )
                        
                        scale_results[eval_task] = {
                            'accuracy': eval_result['accuracy'],
                            'total_questions': eval_result['total']
                        }
                        
                    results['splits'][split][target_task][scale] = scale_results
                    
        # Compute summary statistics
        results['summary'] = self._compute_interference_summary(results)
        
        return results
        
    def _compute_interference_summary(self, results: Dict) -> Dict:
        """Compute summary statistics for cross-task interference."""
        summary = {
            'on_target_effect': {},  # How steering affects target task
            'cross_task_effect': {}, # How steering affects other tasks
            'best_scales': {}        # Best steering scale per task
        }
        
        tasks = ['count', 'color', 'shape']
        
        for target_task in tasks:
            target_accs = []
            cross_accs = []
            
            for split in results['splits']:
                if target_task not in results['splits'][split]:
                    continue
                    
                for scale in results['splits'][split][target_task]:
                    scale_results = results['splits'][split][target_task][scale]
                    
                    # Target task accuracy
                    if target_task in scale_results:
                        target_accs.append(scale_results[target_task]['accuracy'])
                        
                    # Cross-task accuracies
                    for eval_task in tasks:
                        if eval_task != target_task and eval_task in scale_results:
                            cross_accs.append(scale_results[eval_task]['accuracy'])
                            
            # Compute averages
            summary['on_target_effect'][target_task] = {
                'mean_accuracy': np.mean(target_accs) if target_accs else 0.0,
                'std_accuracy': np.std(target_accs) if target_accs else 0.0
            }
            
            summary['cross_task_effect'][target_task] = {
                'mean_accuracy': np.mean(cross_accs) if cross_accs else 0.0,
                'std_accuracy': np.std(cross_accs) if cross_accs else 0.0
            }
            
        return summary
        
    def save_results(self, results: Dict, output_path: str):
        """Save evaluation results to JSON."""
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Add metadata
        results['metadata'] = {
            'model_name': self.model_name,
            'dataset_dir': str(self.dataset_dir),
            'task_vectors_dir': str(self.task_vectors_dir),
            'total_questions': len(self.questions),
            'total_images': len(self.images)
        }
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate steering on visual tasks")
    parser.add_argument("--model", type=str, default="google/paligemma2-3b-mix-224",
                       help="Model name")
    parser.add_argument("--dataset_dir", type=str, default="data",
                       help="Dataset directory") 
    parser.add_argument("--task_vectors_dir", type=str, default="task_vectors",
                       help="Task vectors directory")
    parser.add_argument("--output_file", type=str, default="results/steering_evaluation.json",
                       help="Output file for results")
    parser.add_argument("--max_questions", type=int, default=100,
                       help="Max questions per evaluation")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    parser.add_argument("--eval_baseline", action="store_true",
                       help="Include baseline evaluation")
    parser.add_argument("--eval_cross_task", action="store_true", 
                       help="Evaluate cross-task interference")
    parser.add_argument("--steering_scales", type=float, nargs="+", 
                       default=[0.5, 1.0, 2.0],
                       help="Steering scales to test")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = SteeringEvaluator(
        model_name=args.model,
        dataset_dir=args.dataset_dir,
        task_vectors_dir=args.task_vectors_dir,
        device=args.device
    )
    
    all_results = {}
    
    # Baseline evaluation
    if args.eval_baseline:
        print("=== Baseline Evaluation ===")
        baseline_results = {}
        
        for split in ['A', 'B', 'C']:  # Factor-isolated, mixed, stress
            for task in ['count', 'color', 'shape']:
                questions = evaluator.get_questions_by_split_and_task(split, task)
                if len(questions) > 0:
                    result = evaluator.evaluate_baseline(questions, args.max_questions)
                    
                    if split not in baseline_results:
                        baseline_results[split] = {}
                    baseline_results[split][task] = result
                    
                    print(f"Split {split}, Task {task}: {result['accuracy']:.3f} "
                          f"({result['correct']}/{result['total']})")
                    
        all_results['baseline'] = baseline_results
        
    # Steering evaluation
    print("\n=== Steering Evaluation ===")
    steering_results = {}
    
    for scale in args.steering_scales:
        steering_results[scale] = {}
        
        for split in ['A', 'B', 'C']:
            for task in ['count', 'color', 'shape']:
                questions = evaluator.get_questions_by_split_and_task(split, task)
                if len(questions) > 0 and task in evaluator.task_vectors:
                    result = evaluator.evaluate_with_steering(
                        questions, task, scale, args.max_questions
                    )
                    
                    if split not in steering_results[scale]:
                        steering_results[scale][split] = {}
                    steering_results[scale][split][task] = result
                    
                    print(f"Scale {scale}, Split {split}, Task {task}: "
                          f"{result['accuracy']:.3f} ({result['correct']}/{result['total']})")
                    
    all_results['steering'] = steering_results
    
    # Cross-task interference
    if args.eval_cross_task:
        print("\n=== Cross-Task Interference ===")
        interference_results = evaluator.evaluate_cross_task_interference(
            max_questions_per_task=args.max_questions // 2,
            steering_scales=args.steering_scales
        )
        all_results['cross_task_interference'] = interference_results
        
        # Print summary
        summary = interference_results['summary']
        for task in ['count', 'color', 'shape']:
            if task in summary['on_target_effect']:
                on_target = summary['on_target_effect'][task]['mean_accuracy']
                cross_task = summary['cross_task_effect'][task]['mean_accuracy']
                print(f"{task.capitalize()} steering: "
                      f"on-target={on_target:.3f}, cross-task={cross_task:.3f}")
    
    # Save results
    evaluator.save_results(all_results, args.output_file)
    
    print(f"\n✅ Steering evaluation complete!")
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    main()
