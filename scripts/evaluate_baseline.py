#!/usr/bin/env python3
"""
Baseline Evaluation Script for MLLM Task-Vector Dataset

This script evaluates baseline accuracies for vision-language models 
(like PaliGemma) on the generated shapes dataset.
"""

import json
import os
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from tqdm import tqdm

class BaselineEvaluator:
    def __init__(self, model_name: str, dataset_dir: str, device: str = "auto"):
        """
        Initialize the evaluator with model and dataset paths.
        
        Args:
            model_name: HuggingFace model name (e.g., "google/paligemma2-3b-mix-224")
            dataset_dir: Path to the dataset directory containing manifests and images
            device: Device to run on ("auto", "cuda", "cpu")
        """
        self.model_name = model_name
        self.dataset_dir = Path(dataset_dir)
        
        # Auto-detect device if not specified
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        print(f"Loading model: {model_name}")
        
        # Load model and processor
        try:
            self.processor = AutoProcessor.from_pretrained(model_name)
            
            # Load model with proper device handling
            if self.device == "cuda":
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
            else:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32
                )
                self.model = self.model.to(self.device)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            raise
            
        # Load dataset
        self.load_dataset()
        
    def load_dataset(self):
        """Load the dataset manifests."""
        images_file = self.dataset_dir / "manifests" / "images.jsonl"
        questions_file = self.dataset_dir / "manifests" / "questions.jsonl"
        
        if not images_file.exists():
            raise FileNotFoundError(f"Images manifest not found: {images_file}")
        if not questions_file.exists():
            raise FileNotFoundError(f"Questions manifest not found: {questions_file}")
            
        # Load images metadata
        print("Loading images metadata...")
        self.images = {}
        with open(images_file, 'r') as f:
            for line in f:
                img_data = json.loads(line)
                self.images[img_data['image_id']] = img_data
                
        # Load questions
        print("Loading questions...")
        self.questions = []
        with open(questions_file, 'r') as f:
            for line in f:
                question_data = json.loads(line)
                self.questions.append(question_data)
                
        print(f"Loaded {len(self.images)} images and {len(self.questions)} questions")
        
        # Organize questions by split and task
        self.questions_by_split = defaultdict(list)
        self.questions_by_task = defaultdict(list)
        self.questions_by_split_task = defaultdict(lambda: defaultdict(list))
        
        for q in self.questions:
            split = q['split']
            task = q['task']
            self.questions_by_split[split].append(q)
            self.questions_by_task[task].append(q)
            self.questions_by_split_task[split][task].append(q)
            
        print("Dataset organization:")
        for split in sorted(self.questions_by_split.keys()):
            print(f"  Split {split}: {len(self.questions_by_split[split])} questions")
            for task in sorted(self.questions_by_split_task[split].keys()):
                count = len(self.questions_by_split_task[split][task])
                print(f"    Task {task}: {count} questions")
                
    def load_image(self, image_id: str) -> Image.Image:
        """Load an image by ID."""
        if image_id not in self.images:
            raise ValueError(f"Image {image_id} not found in dataset")
            
        img_info = self.images[image_id]
        img_path = self.dataset_dir / img_info['render']['file']
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image file not found: {img_path}")
            
        return Image.open(img_path).convert('RGB')
        
    def generate_response(self, image: Image.Image, prompt: str) -> str:
        """Generate a response from the model for a given image and prompt."""
        try:
            # Prepare inputs
            inputs = self.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate response
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=10,  # Short responses expected
                    do_sample=False,    # Deterministic for baseline
                    pad_token_id=self.processor.tokenizer.eos_token_id
                )
                
            # Decode response
            response = self.processor.decode(
                generated_ids[0][inputs['input_ids'].shape[1]:],  # Only new tokens
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return ""
            
    def extract_answer(self, response: str, task: str, answer_space: List[str]) -> Optional[str]:
        """
        Extract the answer from model response based on task type.
        
        For our dataset:
        - count: extract first digit
        - color/shape: extract first word that matches answer space
        """
        response = response.lower().strip()
        
        if task == "count":
            # Extract first digit
            for char in response:
                if char.isdigit():
                    return char
            return None
            
        elif task in ["color", "shape"]:
            # Extract first word that matches answer space
            words = response.split()
            answer_space_lower = [ans.lower() for ans in answer_space]
            
            for word in words:
                # Clean word of punctuation
                clean_word = ''.join(c for c in word if c.isalnum())
                if clean_word in answer_space_lower:
                    # Return original case from answer_space
                    idx = answer_space_lower.index(clean_word)
                    return answer_space[idx]
                    
            # If no exact match, try partial matching
            for word in words:
                clean_word = ''.join(c for c in word if c.isalnum())
                for i, ans in enumerate(answer_space_lower):
                    if ans in clean_word or clean_word in ans:
                        return answer_space[i]
                        
            return None
            
        return None
        
    def evaluate_questions(self, questions: List[Dict], max_samples: Optional[int] = None) -> Dict:
        """Evaluate a list of questions and return results."""
        if max_samples:
            questions = questions[:max_samples]
            
        results = {
            'total': len(questions),
            'correct': 0,
            'no_answer': 0,
            'details': []
        }
        
        print(f"Evaluating {len(questions)} questions...")
        
        for i, question in enumerate(tqdm(questions, desc="Processing")):
            try:
                # Load image
                image = self.load_image(question['image_id'])
                
                # Generate response
                response = self.generate_response(image, question['prompt_text'])
                
                # Extract answer
                extracted = self.extract_answer(
                    response, 
                    question['task'], 
                    question['answer_space']
                )
                
                # Check correctness
                is_correct = extracted == question['ground_truth']
                if is_correct:
                    results['correct'] += 1
                    
                if extracted is None:
                    results['no_answer'] += 1
                    
                # Store details
                detail = {
                    'image_id': question['image_id'],
                    'task': question['task'],
                    'prompt': question['prompt_text'],
                    'ground_truth': question['ground_truth'],
                    'response': response,
                    'extracted': extracted,
                    'correct': is_correct
                }
                results['details'].append(detail)
                
                # Print progress occasionally
                if (i + 1) % 50 == 0:
                    acc = results['correct'] / (i + 1) * 100
                    print(f"  Progress: {i+1}/{len(questions)}, Accuracy: {acc:.1f}%")
                    
            except Exception as e:
                print(f"Error processing question {i}: {e}")
                # Add failed case to details
                results['details'].append({
                    'image_id': question.get('image_id', 'unknown'),
                    'task': question.get('task', 'unknown'),
                    'error': str(e),
                    'correct': False
                })
                
        return results
        
    def evaluate_all(self, max_samples_per_task: Optional[int] = None, 
                    splits_to_eval: Optional[List[str]] = None) -> Dict:
        """
        Evaluate all questions in the dataset.
        
        Args:
            max_samples_per_task: Limit samples per task for quick testing
            splits_to_eval: List of splits to evaluate (default: all)
        """
        if splits_to_eval is None:
            splits_to_eval = list(self.questions_by_split.keys())
            
        all_results = {}
        
        for split in splits_to_eval:
            print(f"\n=== Evaluating Split {split} ===")
            split_results = {}
            
            for task in sorted(self.questions_by_split_task[split].keys()):
                print(f"\n--- Task: {task} ---")
                task_questions = self.questions_by_split_task[split][task]
                
                task_results = self.evaluate_questions(task_questions, max_samples_per_task)
                split_results[task] = task_results
                
                # Print task summary
                accuracy = task_results['correct'] / task_results['total'] * 100
                no_ans_pct = task_results['no_answer'] / task_results['total'] * 100
                print(f"Task {task} Results:")
                print(f"  Total: {task_results['total']}")
                print(f"  Correct: {task_results['correct']} ({accuracy:.1f}%)")
                print(f"  No answer: {task_results['no_answer']} ({no_ans_pct:.1f}%)")
                
            all_results[split] = split_results
            
        return all_results
        
    def print_summary(self, results: Dict):
        """Print a summary of evaluation results."""
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        
        for split, split_results in results.items():
            print(f"\nSplit {split}:")
            print("-" * 40)
            
            split_totals = {'total': 0, 'correct': 0, 'no_answer': 0}
            
            for task, task_results in split_results.items():
                accuracy = task_results['correct'] / task_results['total'] * 100
                no_ans_pct = task_results['no_answer'] / task_results['total'] * 100
                
                print(f"  {task:8s}: {accuracy:5.1f}% correct "
                      f"({task_results['correct']:3d}/{task_results['total']:3d}), "
                      f"{no_ans_pct:4.1f}% no answer")
                      
                split_totals['total'] += task_results['total']
                split_totals['correct'] += task_results['correct']
                split_totals['no_answer'] += task_results['no_answer']
                
            # Split overall
            if split_totals['total'] > 0:
                overall_acc = split_totals['correct'] / split_totals['total'] * 100
                overall_no_ans = split_totals['no_answer'] / split_totals['total'] * 100
                print(f"  {'OVERALL':8s}: {overall_acc:5.1f}% correct "
                      f"({split_totals['correct']:3d}/{split_totals['total']:3d}), "
                      f"{overall_no_ans:4.1f}% no answer")
                      
    def save_results(self, results: Dict, output_file: str):
        """Save detailed results to a JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Prepare results for JSON serialization
        results_to_save = {
            'model_name': self.model_name,
            'dataset_dir': str(self.dataset_dir),
            'device': self.device,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': results
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
            
        print(f"\nDetailed results saved to: {output_path}")
        
    def analyze_errors(self, results: Dict, max_examples: int = 5):
        """Analyze common error patterns."""
        print("\n" + "="*60)
        print("ERROR ANALYSIS")
        print("="*60)
        
        for split, split_results in results.items():
            print(f"\nSplit {split}:")
            print("-" * 40)
            
            for task, task_results in split_results.items():
                print(f"\nTask: {task}")
                
                # Collect errors
                errors = [d for d in task_results['details'] if not d.get('correct', False)]
                if not errors:
                    print("  No errors!")
                    continue
                    
                # Analyze error types
                error_types = defaultdict(list)
                for error in errors:
                    if 'error' in error:
                        error_types['processing_error'].append(error)
                    elif error.get('extracted') is None:
                        error_types['no_answer_extracted'].append(error)
                    else:
                        error_types['wrong_answer'].append(error)
                        
                for error_type, error_list in error_types.items():
                    print(f"  {error_type}: {len(error_list)} cases")
                    
                    # Show examples
                    for i, error in enumerate(error_list[:max_examples]):
                        if error_type == 'wrong_answer':
                            print(f"    Ex{i+1}: GT='{error['ground_truth']}', "
                                  f"Got='{error['extracted']}', "
                                  f"Response='{error['response'][:50]}...'")
                        elif error_type == 'no_answer_extracted':
                            print(f"    Ex{i+1}: GT='{error['ground_truth']}', "
                                  f"Response='{error['response'][:50]}...'")


def main():
    parser = argparse.ArgumentParser(description="Evaluate baseline model on shapes dataset")
    parser.add_argument("--model", type=str, default="google/paligemma2-3b-mix-224",
                       help="HuggingFace model name")
    parser.add_argument("--dataset_dir", type=str, default="data",
                       help="Dataset directory path")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["auto", "cuda", "cpu"],
                       help="Device to use")
    parser.add_argument("--splits", type=str, nargs="+", default=None,
                       help="Splits to evaluate (default: all)")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Max samples per task (for testing)")
    parser.add_argument("--output", type=str, default="results/baseline_results.json",
                       help="Output file for detailed results")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test with limited samples")
    
    args = parser.parse_args()
    
    # Quick test mode
    if args.quick:
        args.max_samples = 10
        if args.splits is None:
            args.splits = ['A']  # Only test Split A
        print("Quick test mode: Limited samples and Split A only")
    
    # Create evaluator
    evaluator = BaselineEvaluator(
        model_name=args.model,
        dataset_dir=args.dataset_dir,
        device=args.device
    )
    
    # Run evaluation
    print(f"\nStarting evaluation...")
    results = evaluator.evaluate_all(
        max_samples_per_task=args.max_samples,
        splits_to_eval=args.splits
    )
    
    # Print summary
    evaluator.print_summary(results)
    
    # Analyze errors
    evaluator.analyze_errors(results)
    
    # Save results
    evaluator.save_results(results, args.output)
    
    print(f"\nEvaluation complete!")


if __name__ == "__main__":
    main()
