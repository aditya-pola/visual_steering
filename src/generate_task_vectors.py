"""
Generate task vectors for multimodal LLMs using our visual dataset.

This script generates contrastive activation vectors for count, color, and shape tasks
using the factor-isolated dataset (Split A) with matched question triples.
"""

import json
import torch
import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from PIL import Image
from tqdm import tqdm
import numpy as np

from .paligemma_wrapper import PaliGemmaWrapper


class TaskVectorGenerator:
    """
    Generator for task-specific steering vectors using our visual dataset.
    """
    
    def __init__(self, model_name: str, dataset_dir: str, device: str = "auto"):
        self.model_name = model_name
        self.dataset_dir = Path(dataset_dir)
        self.device = device
        
        # Load MLLM wrapper
        print(f"Initializing {model_name}...")
        self.model = PaliGemmaWrapper(model_name, device)
        
        # Load dataset
        self.images = {}
        self.questions = []
        self.load_dataset()
        
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
                
        # Load questions (only Split A for vector generation)
        with open(questions_file, 'r') as f:
            for line in f:
                question_data = json.loads(line)
                if question_data['split'] == 'A':  # Only factor-isolated split
                    self.questions.append(question_data)
                    
        print(f"Loaded {len(self.images)} images and {len(self.questions)} questions from Split A")
        
    def get_matched_triples(self, max_images: Optional[int] = None) -> List[Dict]:
        """
        Get matched question triples (count, color, shape) for the same images.
        
        Args:
            max_images: Maximum number of images to process (None = all)
            
        Returns:
            List of dictionaries, each containing count/color/shape questions for same image
        """
        # Group questions by image_id
        questions_by_image = {}
        for q in self.questions:
            image_id = q['image_id']
            if image_id not in questions_by_image:
                questions_by_image[image_id] = {}
            questions_by_image[image_id][q['task']] = q
            
        # Filter for complete triples (count + color + shape)
        complete_triples = []
        for image_id, questions in questions_by_image.items():
            if all(task in questions for task in ['count', 'color', 'shape']):
                triple = {
                    'image_id': image_id,
                    'count': questions['count'],
                    'color': questions['color'], 
                    'shape': questions['shape']
                }
                complete_triples.append(triple)
                
        # Limit if requested
        if max_images is not None:
            complete_triples = complete_triples[:max_images]
            
        print(f"Found {len(complete_triples)} complete triples (count+color+shape)")
        return complete_triples
        
    def load_image(self, image_id: str) -> Image.Image:
        """Load an image by ID."""
        if image_id not in self.images:
            raise ValueError(f"Image {image_id} not found")
            
        img_info = self.images[image_id]
        img_path = self.dataset_dir / img_info['render']['file']
        
        return Image.open(img_path).convert('RGB')
        
    def extract_activations_for_triple(self, triple: Dict, 
                                     layer_names: Optional[List[str]] = None,
                                     position_index: int = -2) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Extract activations for a matched triple (count, color, shape).
        
        Args:
            triple: Triple dictionary from get_matched_triples()
            layer_names: Layers to extract from (None = all)
            position_index: Token position to extract activations from
            
        Returns:
            Dictionary mapping task -> layer_name -> activation tensor
        """
        image = self.load_image(triple['image_id'])
        activations = {'count': {}, 'color': {}, 'shape': {}}
        
        for task in ['count', 'color', 'shape']:
            prompt = triple[task]['prompt_text']
            task_activations = self.model.get_activations(
                image, prompt, layer_names, position_index
            )
            activations[task] = task_activations
            
        return activations
        
    def generate_task_vectors(self, max_images: int = 200, 
                            layer_names: Optional[List[str]] = None,
                            position_index: int = -2) -> Dict[str, Dict[str, torch.Tensor]]:
        """
        Generate task vectors for count, color, and shape.
        
        Uses the contrastive approach:
        - task_vector = mean(task_activations) - mean(neutral_activations)
        
        Args:
            max_images: Maximum images to process
            layer_names: Layers to generate vectors for (None = all LM layers)
            position_index: Token position to extract from (-2 = second-to-last)
            
        Returns:
            Dictionary mapping task -> layer_name -> steering vector
        """
        if layer_names is None:
            layer_names = self.model.language_model_layers
            
        print(f"Generating task vectors for layers: {len(layer_names)}")
        print(f"Using position index: {position_index}")
        
        # Get matched triples
        triples = self.get_matched_triples(max_images)
        
        # Collect activations for each task
        task_activations = {
            'count': {layer: [] for layer in layer_names},
            'color': {layer: [] for layer in layer_names},
            'shape': {layer: [] for layer in layer_names}
        }
        
        print("Extracting activations...")
        for triple in tqdm(triples, desc="Processing triples"):
            try:
                activations = self.extract_activations_for_triple(
                    triple, layer_names, position_index
                )
                
                # Store activations for each task and layer
                for task in ['count', 'color', 'shape']:
                    for layer_name in layer_names:
                        if layer_name in activations[task]:
                            task_activations[task][layer_name].append(
                                activations[task][layer_name]
                            )
                            
            except Exception as e:
                print(f"Error processing triple {triple['image_id']}: {e}")
                continue
                
        # Compute task vectors using neutral baseline
        print("Computing contrastive task vectors...")
        task_vectors = {}
        
        for task in ['count', 'color', 'shape']:
            task_vectors[task] = {}
            
            for layer_name in layer_names:
                if len(task_activations[task][layer_name]) > 0:
                    # Stack activations
                    task_acts = torch.stack(task_activations[task][layer_name])
                    
                    # For contrastive approach, we need a baseline
                    # We'll use the mean activation across all tasks as neutral
                    all_acts = []
                    for t in ['count', 'color', 'shape']:
                        all_acts.extend(task_activations[t][layer_name])
                    
                    if len(all_acts) > 0:
                        neutral_acts = torch.stack(all_acts).mean(dim=0)
                        task_mean = task_acts.mean(dim=0)
                        
                        # Task vector = task_mean - neutral_mean
                        task_vectors[task][layer_name] = task_mean - neutral_acts
                        
        # Print statistics
        for task in ['count', 'color', 'shape']:
            valid_layers = len(task_vectors[task])
            print(f"Generated {valid_layers} vectors for task '{task}'")
            
        return task_vectors
        
    def save_task_vectors(self, task_vectors: Dict[str, Dict[str, torch.Tensor]], 
                         output_dir: str):
        """Save generated task vectors to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving task vectors to {output_path}")
        
        for task, layer_vectors in task_vectors.items():
            task_dir = output_path / task
            task_dir.mkdir(exist_ok=True)
            
            for layer_name, vector in layer_vectors.items():
                # Create safe filename from layer name
                safe_layer_name = layer_name.replace('.', '_').replace('/', '_')
                vector_path = task_dir / f"vector_{safe_layer_name}.pt"
                
                torch.save(vector, vector_path)
                
            print(f"Saved {len(layer_vectors)} vectors for task '{task}'")
            
        # Save metadata
        metadata = {
            'model_name': self.model_name,
            'dataset_dir': str(self.dataset_dir),
            'tasks': list(task_vectors.keys()),
            'layers': {
                task: list(layer_vectors.keys()) 
                for task, layer_vectors in task_vectors.items()
            },
            'vector_shape': {
                task: {
                    layer_name: list(vector.shape)
                    for layer_name, vector in layer_vectors.items()
                }
                for task, layer_vectors in task_vectors.items()
            }
        }
        
        with open(output_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
            
        print(f"Saved metadata to {output_path / 'metadata.json'}")
        
    def analyze_vectors(self, task_vectors: Dict[str, Dict[str, torch.Tensor]]):
        """Analyze properties of generated task vectors."""
        print("\n=== Task Vector Analysis ===")
        
        for task in ['count', 'color', 'shape']:
            if task not in task_vectors:
                continue
                
            print(f"\nTask: {task}")
            layer_vectors = task_vectors[task]
            
            if len(layer_vectors) == 0:
                print("  No vectors generated")
                continue
                
            # Vector statistics
            norms = [torch.norm(vec).item() for vec in layer_vectors.values()]
            dims = [vec.shape[0] for vec in layer_vectors.values()]
            
            print(f"  Layers: {len(layer_vectors)}")
            print(f"  Vector dimension: {dims[0]} (consistent: {len(set(dims)) == 1})")
            print(f"  Norm range: [{min(norms):.3f}, {max(norms):.3f}]")
            print(f"  Mean norm: {np.mean(norms):.3f}")
            
        # Cross-task similarities
        print("\n=== Cross-Task Similarities ===")
        tasks = ['count', 'color', 'shape']
        
        # Compare first layer vectors as example
        first_layer = next(iter(task_vectors['count'].keys())) if 'count' in task_vectors else None
        
        if first_layer and all(first_layer in task_vectors[t] for t in tasks):
            print(f"Similarities for layer: {first_layer}")
            
            for i, task1 in enumerate(tasks):
                for task2 in tasks[i+1:]:
                    vec1 = task_vectors[task1][first_layer]
                    vec2 = task_vectors[task2][first_layer]
                    
                    # Cosine similarity
                    cos_sim = torch.cosine_similarity(vec1, vec2, dim=0)
                    print(f"  {task1} vs {task2}: {cos_sim.item():.3f}")


def main():
    parser = argparse.ArgumentParser(description="Generate task vectors for visual MLLMs")
    parser.add_argument("--model", type=str, default="google/paligemma2-3b-mix-224",
                       help="Model name")
    parser.add_argument("--dataset_dir", type=str, default="data", 
                       help="Dataset directory")
    parser.add_argument("--output_dir", type=str, default="task_vectors",
                       help="Output directory for vectors")
    parser.add_argument("--max_images", type=int, default=200,
                       help="Maximum images to process")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to use")
    parser.add_argument("--layers", type=str, nargs="+", default=None,
                       help="Specific layers to generate vectors for")
    parser.add_argument("--position_index", type=int, default=-2,
                       help="Token position to extract activations from")
    
    args = parser.parse_args()
    
    # Initialize generator
    generator = TaskVectorGenerator(
        model_name=args.model,
        dataset_dir=args.dataset_dir,
        device=args.device
    )
    
    # Generate task vectors
    task_vectors = generator.generate_task_vectors(
        max_images=args.max_images,
        layer_names=args.layers,
        position_index=args.position_index
    )
    
    # Analyze vectors
    generator.analyze_vectors(task_vectors)
    
    # Save vectors
    generator.save_task_vectors(task_vectors, args.output_dir)
    
    print(f"\n✅ Task vector generation complete!")
    print(f"Output saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
