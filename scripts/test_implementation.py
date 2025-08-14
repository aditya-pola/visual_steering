#!/usr/bin/env python3
"""
Test script for the visual steering implementation.

This script performs basic validation of the core components.
"""

import torch
import json
import sys
from pathlib import Path
from PIL import Image

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
sys.path.append(str(project_root / "src"))

from src.paligemma_wrapper import PaliGemmaWrapper, get_cuda0_device_config
from src.generate_task_vectors import TaskVectorGenerator


def test_basic_model_loading():
    """Test basic model loading and inference."""
    print("=== Testing Basic Model Loading ===")
    
    try:
        # Load model with specific CUDA 0 device
        device = get_cuda0_device_config()
        model = PaliGemmaWrapper("google/paligemma2-3b-mix-224", device=device)
        print("✅ Model loaded successfully")
        
        # Create a simple test image
        test_image = Image.new('RGB', (224, 224), color='red')
        test_prompt = "What color is this?"
        
        # Test basic generation
        response = model.generate(test_image, test_prompt, max_new_tokens=5)
        print(f"✅ Basic generation works: '{response}'")
        
        return True, model
        
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return False, None


def test_activation_extraction(model):
    """Test activation extraction."""
    print("\n=== Testing Activation Extraction ===")
    
    try:
        test_image = Image.new('RGB', (224, 224), color='blue')
        test_prompt = "What color is this?"
        
        # Get activations from first few layers
        layer_names = model.language_model_layers[:3]  # Test first 3 layers
        activations = model.get_activations(test_image, test_prompt, layer_names)
        
        print(f"✅ Extracted activations from {len(activations)} layers")
        for layer_name, activation in activations.items():
            print(f"  {layer_name}: shape {activation.shape}")
            
        return True, activations
        
    except Exception as e:
        print(f"❌ Activation extraction failed: {e}")
        return False, None


def test_steering_application(model, activations):
    """Test steering vector application."""
    print("\n=== Testing Steering Application ===")
    
    try:
        test_image = Image.new('RGB', (224, 224), color='green')
        test_prompt = "What color is this?"
        
        # Use first activation as a dummy steering vector, but scale it down significantly
        layer_name = next(iter(activations.keys()))
        dummy_vector = activations[layer_name] * 0.001  # Very small perturbation
        
        steering_vectors = {layer_name: dummy_vector}
        
        print(f"Using layer: {layer_name}")
        print(f"Vector dtype: {dummy_vector.dtype}, shape: {dummy_vector.shape}")
        print(f"Model parameter dtype: {next(model.model.parameters()).dtype}")
        
        # Generate with steering
        response = model.generate_with_steering(
            image=test_image,
            prompt=test_prompt,
            steering_vectors=steering_vectors,
            steering_scale=0.1,
            max_new_tokens=5
        )
        
        print(f"✅ Steering generation works: '{response}'")
        return True
        
    except Exception as e:
        print(f"❌ Steering application failed: {e}")
        import traceback
        traceback.print_exc()
        return False
def test_dataset_loading():
    """Test dataset loading if dataset exists."""
    print("\n=== Testing Dataset Loading ===")
    
    dataset_dir = Path("data")
    if not dataset_dir.exists():
        print("⏭️ No dataset found, skipping dataset tests")
        return True
        
    try:
        # Test TaskVectorGenerator loading
        device = get_cuda0_device_config()
        generator = TaskVectorGenerator(
            model_name="google/paligemma2-3b-mix-224",
            dataset_dir="data",
            device=device
        )
        
        print(f"✅ Dataset loaded: {len(generator.images)} images, {len(generator.questions)} questions")
        
        # Test getting matched triples
        triples = generator.get_matched_triples(max_images=5)
        print(f"✅ Found {len(triples)} matched triples")
        
        if len(triples) > 0:
            # Test loading an image
            image = generator.load_image(triples[0]['image_id'])
            print(f"✅ Image loading works: {image.size}")
            
        return True
        
    except Exception as e:
        print(f"❌ Dataset loading failed: {e}")
        return False


def main():
    """Run all tests."""
    print("🧪 Running Visual Steering Tests")
    print("=" * 50)
    
    # Test 1: Basic model loading
    model_success, model = test_basic_model_loading()
    if not model_success:
        print("❌ Critical failure: Cannot load model")
        return 1
        
    # Test 2: Activation extraction  
    activation_success, activations = test_activation_extraction(model)
    if not activation_success:
        print("❌ Critical failure: Cannot extract activations")
        return 1
        
    # Test 3: Steering application
    steering_success = test_steering_application(model, activations)
    if not steering_success:
        print("❌ Critical failure: Cannot apply steering")
        return 1
        
    # Test 4: Dataset loading (optional)
    dataset_success = test_dataset_loading()
    
    print("\n" + "=" * 50)
    
    if all([model_success, activation_success, steering_success, dataset_success]):
        print("🎉 All tests passed!")
        return 0
    else:
        print("⚠️ Some tests failed, but core functionality works")
        return 0


if __name__ == "__main__":
    sys.exit(main())
