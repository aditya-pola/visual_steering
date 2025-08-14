#!/usr/bin/env python3
"""
Test steering vector performance by adding vectors during generation without question prompts.
Only the image is provided, and steering vectors are applied during first token generation.
"""
import sys
import json
import torch
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.paligemma_wrapper import PaliGemmaWrapper
from src.generate_task_vectors import TaskVectorGenerator


class SteeringVectorTester:
    """Test steering vectors by applying them during generation without question prompts."""
    
    def __init__(self, model_name: str, dataset_dir: str, vectors_dir: str, device: str = "auto"):
        self.model_name = model_name
        self.dataset_dir = Path(dataset_dir)
        self.vectors_dir = Path(vectors_dir)
        self.device = device
        
        print(f"Initializing {model_name}...")
        self.model = PaliGemmaWrapper(model_name, device)
        
        # Load dataset
        self.images = {}
        self.questions = []
        self.load_dataset()
        
        # Load task vectors
        self.task_vectors = {}
        self.load_task_vectors()
        
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
                
        # Load questions (Split A for testing - use different samples than training)
        with open(questions_file, 'r') as f:
            for line in f:
                question_data = json.loads(line)
                if question_data['split'] == 'A':  # Use Split A but different samples
                    self.questions.append(question_data)
                    
        print(f"Loaded {len(self.images)} images and {len(self.questions)} questions from Split A")
        
    def load_task_vectors(self):
        """Load pre-computed task vectors."""
        metadata_file = self.vectors_dir / "metadata.json"
        
        if not metadata_file.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}")
            
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        print("Loading task vectors...")
        
        for task in metadata['tasks']:
            self.task_vectors[task] = {}
            task_dir = self.vectors_dir / task
            
            for layer_name in metadata['layers'][task]:
                # Convert layer name to safe filename
                safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
                vector_file = task_dir / f"vector_{safe_layer_name}.pt"
                
                if vector_file.exists():
                    vector = torch.load(vector_file, map_location='cpu')
                    self.task_vectors[task][layer_name] = vector
                    
        print(f"Loaded task vectors for: {list(self.task_vectors.keys())}")
        
    def load_image(self, image_id: str) -> Image.Image:
        """Load an image by ID."""
        if image_id not in self.images:
            raise ValueError(f"Image {image_id} not found")
            
        img_info = self.images[image_id]
        img_path = self.dataset_dir / img_info['render']['file']
        
        return Image.open(img_path).convert('RGB')
        
    def generate_steered_response(self, image: Image.Image, task: str, 
                                layers: Optional[List[str]] = None,
                                steering_strength: float = 1.0) -> str:
        """
        Generate response with steering vector applied during generation.
        Only image is provided as input - no question prompt.
        
        Args:
            image: Input image
            task: Task type ('count', 'color', 'shape')
            layers: Specific layers to apply steering (None = all available)
            steering_strength: Multiplier for steering vector strength
            
        Returns:
            Generated text response
        """
        if task not in self.task_vectors:
            raise ValueError(f"Task '{task}' not found in loaded vectors")
            
        # Prepare steering vectors for specified layers
        steering_vectors = {}
        available_layers = list(self.task_vectors[task].keys())
        
        if layers is None:
            layers = available_layers
        else:
            # Filter to only available layers
            layers = [layer for layer in layers if layer in available_layers]
            
        for layer_name in layers:
            if layer_name in self.task_vectors[task]:
                steering_vectors[layer_name] = (
                    self.task_vectors[task][layer_name] * steering_strength
                )
                
        # Generate response with steering (no question prompt - just image)
        prompt = ""  # Empty prompt - only image provided
        
        try:
            response = self.model.generate_with_steering(
                image=image,
                prompt=prompt,
                steering_vectors=steering_vectors,
                steering_scale=steering_strength,
                max_new_tokens=32
            )
            return response.strip()
            
        except Exception as e:
            print(f"Error during steered generation: {e}")
            return f"<ERROR: {str(e)}>"
            
    def generate_baseline_response(self, image: Image.Image) -> str:
        """Generate baseline response without steering (image only)."""
        try:
            response = self.model.generate(
                image=image,
                prompt="",  # Empty prompt - only image
                max_new_tokens=32
            )
            return response.strip()
            
        except Exception as e:
            print(f"Error during baseline generation: {e}")
            return f"<ERROR: {str(e)}>"
            
    def test_steering_performance(self, num_samples: int = 100, 
                                layers_to_test: Optional[List[str]] = None,
                                steering_strengths: List[float] = [0.5, 1.0, 2.0]) -> Dict:
        """
        Test steering vector performance across different tasks and strengths.
        
        Args:
            num_samples: Number of samples to test per task
            layers_to_test: Specific layers to test (None = test different layer groups)
            steering_strengths: Different steering strengths to test
            
        Returns:
            Dictionary with test results
        """
        results = {
            'metadata': {
                'model_name': self.model_name,
                'timestamp': datetime.now().isoformat(),
                'num_samples': num_samples,
                'steering_strengths': steering_strengths,
                'layers_tested': layers_to_test
            },
            'tasks': {}
        }
        
        # Define layer groups to test if not specified
        if layers_to_test is None:
            all_layers = list(self.task_vectors['count'].keys())
            layer_groups = {
                'early': all_layers[:6],      # Layers 0-5
                'middle': all_layers[6:12],   # Layers 6-11  
                'late': all_layers[12:],      # Layers 12-17
                'all': all_layers             # All layers
            }
        else:
            layer_groups = {'specified': layers_to_test}
            
        # Group questions by task
        questions_by_task = {}
        for q in self.questions:
            task = q['task']
            if task not in questions_by_task:
                questions_by_task[task] = []
            questions_by_task[task].append(q)
            
        # Test each task
        for task in ['count', 'color', 'shape']:
            if task not in questions_by_task:
                print(f"No questions found for task: {task}")
                continue
                
            print(f"\nTesting task: {task}")
            results['tasks'][task] = {}
            
            # Sample questions for this task (skip first 200 used for training)
            task_questions = questions_by_task[task]
            
            # Group by image_id to get complete triples, then skip first 200 images
            images_for_task = {}
            for q in task_questions:
                img_id = q['image_id'] 
                if img_id not in images_for_task:
                    images_for_task[img_id] = []
                images_for_task[img_id].append(q)
            
            # Skip first 200 images (used for training), then take num_samples for testing
            test_images = list(images_for_task.keys())[200:200+num_samples]
            task_questions = []
            for img_id in test_images:
                # Find the question for this task and image
                for q in images_for_task[img_id]:
                    if q['task'] == task:
                        task_questions.append(q)
                        break
            
            # Test different layer groups
            for group_name, layers in layer_groups.items():
                print(f"  Testing layer group: {group_name} ({len(layers)} layers)")
                results['tasks'][task][group_name] = {}
                
                # Test different steering strengths
                for strength in steering_strengths:
                    print(f"    Testing steering strength: {strength}")
                    
                    group_results = {
                        'strength': strength,
                        'layers': layers,
                        'samples': []
                    }
                    
                    for i, question in enumerate(tqdm(task_questions, 
                                                    desc=f"{task}-{group_name}-{strength}")):
                        try:
                            image = self.load_image(question['image_id'])
                            
                            # Generate baseline response
                            baseline_response = self.generate_baseline_response(image)
                            
                            # Generate steered response
                            steered_response = self.generate_steered_response(
                                image, task, layers, strength
                            )
                            
                            # Store results
                            sample_result = {
                                'image_id': question['image_id'],
                                'question': question['prompt_text'],
                                'correct_answer': question.get('ground_truth', question.get('answer', 'unknown')),
                                'baseline_response': baseline_response,
                                'steered_response': steered_response,
                                'task': task
                            }
                            
                            # Add ground truth attributes for analysis
                            if question['image_id'] in self.images:
                                img_info = self.images[question['image_id']]
                                sample_result['ground_truth'] = {
                                    'count': img_info.get('count', 'unknown'),
                                    'color': img_info.get('primary_color', 'unknown'),
                                    'shape': img_info.get('shape', 'unknown')
                                }
                            
                            group_results['samples'].append(sample_result)
                            
                        except Exception as e:
                            print(f"Error processing sample {i}: {e}")
                            continue
                            
                    results['tasks'][task][group_name][f'strength_{strength}'] = group_results
                    
        return results
        
    def save_results(self, results: Dict, output_file: str):
        """Save test results to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"Results saved to: {output_path}")


def main():
    """Main function to run steering vector tests."""
    parser = argparse.ArgumentParser(description='Test steering vector performance')
    parser.add_argument('--model', default='google/paligemma-3b-pt-224',
                       help='Model name to test')
    parser.add_argument('--dataset-dir', default='data',
                       help='Dataset directory')
    parser.add_argument('--vectors-dir', default='task_vectors',
                       help='Task vectors directory')
    parser.add_argument('--num-samples', type=int, default=50,
                       help='Number of samples to test per task')
    parser.add_argument('--output-file', 
                       default='results/steering_test_results.json',
                       help='Output file for results')
    parser.add_argument('--device', default='auto',
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    # Convert relative paths to absolute
    project_root = Path(__file__).parent
    dataset_dir = project_root / args.dataset_dir
    vectors_dir = project_root / args.vectors_dir
    output_file = project_root / args.output_file
    
    print("=== Steering Vector Performance Test ===")
    print(f"Model: {args.model}")
    print(f"Dataset: {dataset_dir}")
    print(f"Vectors: {vectors_dir}")
    print(f"Samples per task: {args.num_samples}")
    print(f"Output: {output_file}")
    
    # Initialize tester
    tester = SteeringVectorTester(
        model_name=args.model,
        dataset_dir=str(dataset_dir),
        vectors_dir=str(vectors_dir),
        device=args.device
    )
    
    # Run tests
    print("\nRunning steering vector tests...")
    results = tester.test_steering_performance(
        num_samples=args.num_samples,
        steering_strengths=[0.5, 1.0, 2.0]
    )
    
    # Save results
    tester.save_results(results, str(output_file))
    
    # Print summary
    print(f"\n=== Test Summary ===")
    for task, task_results in results['tasks'].items():
        print(f"Task '{task}': {len(task_results)} layer groups tested")
    
    print(f"Total samples tested: {sum(len(tr) for tr in results['tasks'].values()) * args.num_samples}")
    print("Test completed successfully!")


if __name__ == "__main__":
    main()
