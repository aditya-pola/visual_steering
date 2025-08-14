#!/usr/bin/env python3
"""
Build steering vectors for count, color, and shape from 200 samples.
"""
import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.generate_task_vectors import TaskVectorGenerator

def main():
    """Build steering vectors from 200 samples."""
    print("Building steering vectors for count, color, and shape from 200 samples...")
    
    # Configuration
    model_name = "google/paligemma-3b-pt-224"
    dataset_dir = project_root / "data"
    output_dir = project_root / "task_vectors"
    
    # Create output directory
    output_dir.mkdir(exist_ok=True)
    
    # Initialize generator
    generator = TaskVectorGenerator(
        model_name=model_name,
        dataset_dir=str(dataset_dir),
        device="auto"
    )
    
    # Generate task vectors with 200 samples
    print(f"\nGenerating steering vectors for count, color, and shape tasks...")
    try:
        task_vectors = generator.generate_task_vectors(
            max_images=200,
            layer_names=None,  # Use all language model layers
            position_index=-2  # Second-to-last token position
        )
        
        # Save the vectors
        generator.save_task_vectors(task_vectors, str(output_dir))
        
        # Print summary
        print(f"\n✓ Successfully generated steering vectors!")
        for task in ['count', 'color', 'shape']:
            if task in task_vectors:
                num_layers = len(task_vectors[task])
                print(f"  - {task}: {num_layers} layer vectors")
        
    except Exception as e:
        print(f"✗ Error generating steering vectors: {e}")
        return
    
    print(f"\nSteering vectors saved in: {output_dir}")

if __name__ == "__main__":
    main()
