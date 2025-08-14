"""
PaliGemma-specific implementation of MLLM wrapper for Contrastive Activation Addition.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
from contextlib import contextmanager
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq
from .mllm_wrapper import MLLMWrapper
from utils.nethook import TraceWithEdit, add_vector_to_activations


def get_cuda0_device_config():
    """
    Get device configuration that forces all model components to CUDA 0.
    
    Returns:
        Device string that can be passed to PaliGemmaWrapper
    """
    return "cuda:0"


def get_single_gpu_device_config():
    """
    Get device configuration for single GPU usage (CUDA 0).
    
    Returns:
        Device string that avoids multi-GPU distribution
    """
    return "cuda:0"


class PaliGemmaBlockWrapper(nn.Module):
    """
    Wrapper for PaliGemma language model blocks to enable activation steering.
    Based on the CAA BlockOutputWrapper but adapted for multimodal context.
    """
    
    def __init__(self, block: nn.Module, layer_name: str):
        super().__init__()
        self.block = block
        self.layer_name = layer_name
        
        # Forward all attributes to the wrapped block to maintain compatibility
        self._forwarded_attrs = set()
        
        # Steering configuration
        self.steering_vector = None
        self.steering_multiplier = 1.0
        self.steering_from_position = None
        
        # Activation storage
        self.last_activations = None
        
    def __getattr__(self, name):
        """Forward attribute access to the wrapped block."""
        if name in ['block', 'layer_name', 'steering_vector', 'steering_multiplier', 
                    'steering_from_position', 'last_activations', '_forwarded_attrs']:
            return super().__getattr__(name)
        
        # Forward to wrapped block
        if hasattr(self.block, name):
            attr = getattr(self.block, name)
            self._forwarded_attrs.add(name)
            return attr
            
        return super().__getattr__(name)
        
    def set_steering(self, vector: torch.Tensor, multiplier: float = 1.0, 
                    from_position: Optional[int] = None):
        """Configure steering for this block."""
        self.steering_vector = vector
        self.steering_multiplier = multiplier 
        self.steering_from_position = from_position
        
    def clear_steering(self):
        """Clear steering configuration."""
        self.steering_vector = None
        self.steering_multiplier = 1.0
        self.steering_from_position = None
        
    def forward(self, *args, **kwargs):
        # Forward pass through original block
        output = self.block(*args, **kwargs)
        
        # Store activations
        if isinstance(output, (tuple, list)):
            self.last_activations = output[0].detach()
        else:
            self.last_activations = output.detach()
            
        # Apply steering if configured
        if self.steering_vector is not None:
            position_ids = kwargs.get('position_ids', None)
            
            if isinstance(output, (tuple, list)):
                # Modify the hidden states (first element)
                modified_hidden = add_vector_to_activations(
                    output[0],
                    self.steering_vector * self.steering_multiplier,
                    position_ids,
                    self.steering_from_position
                )
                output = (modified_hidden,) + output[1:]
            else:
                # Single tensor output
                output = add_vector_to_activations(
                    output,
                    self.steering_vector * self.steering_multiplier, 
                    position_ids,
                    self.steering_from_position
                )
                
        return output


class PaliGemmaWrapper(MLLMWrapper):
    """
    PaliGemma-specific implementation of multimodal LLM wrapper.
    
    PaliGemma Architecture:
    - Vision Encoder (SigLIP) processes images  
    - Projection layer aligns vision and text dimensions
    - Language Model (Gemma) generates text from image+text tokens
    
    We focus on steering the Gemma language model component.
    """
    
    def __init__(self, model_name: str = "google/paligemma2-3b-mix-224", device: str = "auto"):
        super().__init__(model_name, device)
        self.wrapped_blocks = {}  # Dict[layer_name, PaliGemmaBlockWrapper]
        self.original_blocks = {}  # Store original blocks for restoration
        
        # Load model and setup
        self.load_model()
        self._setup_language_model_layers()
        self._wrap_language_model_blocks()
        
    @property
    def model_device(self):
        """Get the device where the model is actually located."""
        # For multi-GPU models, get device from the first parameter
        return next(self.model.parameters()).device
        
    def _get_device_map(self):
        """Get device map configuration for model loading."""
        if self.device == "cuda:0" or self.device == "cuda":
            # Force all components to CUDA 0 based on actual model structure
            return {
                'model.vision_tower': 'cuda:0',
                'model.multi_modal_projector': 'cuda:0', 
                'model.language_model': 'cuda:0',
                'lm_head': 'cuda:0'
            }
        elif self.device == "auto":
            return "auto"
        else:
            return None
    
    def load_model(self) -> None:
        """Load PaliGemma model and processor."""
        print(f"Loading PaliGemma model: {self.model_name}")
        
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name)
            
            device_map = self._get_device_map()
            
            # Load model with appropriate precision and device handling
            if device_map is not None:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float16,
                    device_map=device_map
                )
                # Update our device reference to match actual model device
                if isinstance(device_map, dict):
                    self.device = list(device_map.values())[0]  # Use first device
            else:
                self.model = AutoModelForVision2Seq.from_pretrained(
                    self.model_name,
                    torch_dtype=torch.float32
                )
                self.model = self.model.to(self.device)
                
            self.tokenizer = self.processor.tokenizer
            print("PaliGemma model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading PaliGemma model: {e}")
            raise
            
    def get_language_model_layers(self) -> List[str]:
        """Get language model layer names for PaliGemma."""
        layer_names = []
        
        # PaliGemma has the language model under .language_model with direct .layers access
        if hasattr(self.model, 'language_model') and hasattr(self.model.language_model, 'layers'):
            layers = self.model.language_model.layers
            for i in range(len(layers)):
                layer_names.append(f'language_model.layers.{i}')
        
        return layer_names
        
    def _setup_language_model_layers(self):
        """Setup the list of language model layer names."""
        self.language_model_layers = self.get_language_model_layers()
        print(f"Found {len(self.language_model_layers)} language model layers")
        
    def _wrap_language_model_blocks(self):
        """Wrap language model blocks with our steering-capable wrappers."""
        if not hasattr(self.model, 'language_model'):
            print("Warning: Could not find language_model in PaliGemma model")
            return
            
        lm = self.model.language_model
        if not hasattr(lm, 'layers'):
            print("Warning: Could not find layers in language model")
            return
            
        print(f"Wrapping {len(lm.layers)} language model blocks...")
        
        for i, block in enumerate(lm.layers):
            layer_name = f'language_model.layers.{i}'
            
            # Store original block
            self.original_blocks[layer_name] = block
            
            # Create wrapper
            wrapper = PaliGemmaBlockWrapper(block, layer_name)
            self.wrapped_blocks[layer_name] = wrapper
            
            # Replace the block in the model
            lm.layers[i] = wrapper
            
    def _restore_original_blocks(self):
        """Restore original blocks (cleanup method)."""
        if not hasattr(self.model, 'language_model'):
            return
            
        lm = self.model.language_model
        if not hasattr(lm, 'layers'):
            return
            
        for i, layer_name in enumerate(self.language_model_layers):
            if layer_name in self.original_blocks:
                lm.layers[i] = self.original_blocks[layer_name]
                
        self.wrapped_blocks.clear()
        self.original_blocks.clear()
        
    def process_inputs(self, image: Image.Image, text: str) -> Dict[str, torch.Tensor]:
        """Process image and text into model inputs for PaliGemma."""
        # PaliGemma expects specific format
        inputs = self.processor(
            text=text,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        return inputs
        
    def find_text_generation_start(self, input_ids: torch.Tensor) -> int:
        """
        Find where text generation starts in PaliGemma inputs.
        
        PaliGemma prepends image tokens, so text generation starts after:
        - Image tokens
        - Text prompt tokens
        """
        # For PaliGemma, this is complex and depends on the exact tokenization
        # For now, use a simple heuristic: start after the prompt
        # This should be refined based on actual PaliGemma tokenization behavior
        
        # Simple approach: assume text generation starts near the end of input
        seq_len = input_ids.shape[1]
        return max(0, seq_len - 5)  # Start steering from last few tokens
        
    def get_activations(self, image: Image.Image, text: str, 
                       layer_names: Optional[List[str]] = None,
                       position_index: int = -2) -> Dict[str, torch.Tensor]:
        """
        Extract activations from specified layers at a specific position.
        
        Args:
            image: Input image
            text: Input text  
            layer_names: Layer names to extract from
            position_index: Token position to extract (-2 = second-to-last)
            
        Returns:
            Dictionary mapping layer names to activation tensors
        """
        if layer_names is None:
            layer_names = self.language_model_layers
            
        inputs = self.process_inputs(image, text)
        
        # Forward pass and collect activations from wrapped blocks
        with torch.no_grad():
            _ = self.model(**inputs)
            
        activations = {}
        for layer_name in layer_names:
            if layer_name in self.wrapped_blocks:
                block = self.wrapped_blocks[layer_name]
                if block.last_activations is not None:
                    # Extract activation from specific position
                    act = block.last_activations[0, position_index, :]  # [hidden_dim]
                    activations[layer_name] = act.cpu()
                    
        return activations
        
    def set_steering_vector(self, layer_name: str, vector: torch.Tensor, 
                           multiplier: float = 1.0, from_position: Optional[int] = None) -> None:
        """Set steering vector for a specific layer."""
        super().set_steering_vector(layer_name, vector, multiplier, from_position)
        
        # Configure the wrapped block
        if layer_name in self.wrapped_blocks:
            if from_position is None:
                # Auto-detect text generation start
                # For now, use a simple approach
                from_position = -10  # Apply to last 10 tokens
                
            # Ensure vector is on the same device and dtype as the model
            vector_on_device = vector.to(device=self.model_device, dtype=next(self.model.parameters()).dtype)
                
            self.wrapped_blocks[layer_name].set_steering(
                vector_on_device, 
                multiplier, 
                from_position
            )
            
    def clear_steering(self) -> None:
        """Clear all steering vectors."""
        super().clear_steering()
        
        # Clear steering from all wrapped blocks
        for wrapper in self.wrapped_blocks.values():
            wrapper.clear_steering()
            
    def generate_text(self, image: Image.Image, prompt: str, max_new_tokens: int = 50) -> str:
        """Generate text with optional steering applied."""
        inputs = self.process_inputs(image, prompt)
        
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
            
        # Decode only the newly generated tokens
        input_length = inputs['input_ids'].shape[1]  
        new_tokens = generated_ids[0][input_length:]
        response = self.processor.decode(new_tokens, skip_special_tokens=True)
        
        return response.strip()
        
    def generate(self, image: Image.Image, prompt: str, max_new_tokens: int = 10) -> str:
        """
        Generate text response for an image and prompt.
        
        Args:
            image: PIL Image
            prompt: Text prompt
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        inputs = self.processor(text=prompt, images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                pad_token_id=self.processor.tokenizer.eos_token_id
            )
            
        # Decode only the new tokens
        input_len = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][input_len:]
        response = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return response.strip()
        
    def generate_with_steering(self, image: Image.Image, prompt: str, 
                             steering_vectors: Dict[str, torch.Tensor],
                             steering_scale: float = 1.0,
                             max_new_tokens: int = 10) -> str:
        """
        Generate text response with activation steering.
        
        Args:
            image: PIL Image
            prompt: Text prompt
            steering_vectors: Dictionary of layer_name -> steering_vector
            steering_scale: Scaling factor for steering vectors
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        # Set steering vectors
        for layer_name, vector in steering_vectors.items():
            self.set_steering_vector(layer_name, vector, steering_scale)
        
        try:
            # Generate with steering active
            response = self.generate(image, prompt, max_new_tokens)
        finally:
            # Always clear steering after generation
            self.clear_steering()
            
        return response
        
    def compute_task_vectors_from_dataset(self, dataset_path: str, 
                                        task_pairs: List[Tuple[str, str]],
                                        max_samples: int = 100) -> Dict[str, torch.Tensor]:
        """
        Compute task vectors using our visual dataset.
        
        Args:
            dataset_path: Path to dataset directory
            task_pairs: List of (task_name, neutral_name) pairs
            max_samples: Maximum samples to use per task
            
        Returns:
            Dictionary mapping task names to steering vectors
        """
        # This will be implemented to use our generated dataset
        # For now, return empty dict
        return {}
        
    def __del__(self):
        """Cleanup method."""
        if hasattr(self, 'wrapped_blocks'):
            self._restore_original_blocks()
