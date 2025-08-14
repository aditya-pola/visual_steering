"""
Abstract base class for Multimodal Large Language Model wrappers.
Provides a common interface for activation extraction and steering across different MLLM architectures.
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union, Any
from PIL import Image
from transformers import PreTrainedTokenizer, PreTrainedModel
from utils.nethook import TraceDict, ActivationCollector, add_vector_to_activations


class MLLMWrapper(ABC):
    """
    Abstract base class for multimodal language model wrappers.
    
    This class provides a common interface for:
    1. Loading and managing multimodal models  
    2. Processing vision + text inputs
    3. Extracting activations from language model components
    4. Applying steering vectors during inference
    """
    
    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = model_name
        self.device = self._setup_device(device)
        self.model = None
        self.processor = None
        self.tokenizer = None
        
        # Activation management
        self.activation_collector = None
        self.steering_vectors = {}  # Dict[layer_name, torch.Tensor]
        self.steering_multipliers = {}  # Dict[layer_name, float]
        self.steering_positions = {}  # Dict[layer_name, int]
        
        # Model architecture info (to be set by subclasses)
        self.language_model_layers = []  # List of layer names for LM component
        self.vision_encoder_layers = []  # List of layer names for vision component
        
    @abstractmethod
    def load_model(self) -> None:
        """Load the model and processor. To be implemented by subclasses."""
        pass
        
    @abstractmethod  
    def get_language_model_layers(self) -> List[str]:
        """Get the names of language model layers. To be implemented by subclasses."""
        pass
        
    @abstractmethod
    def process_inputs(self, image: Image.Image, text: str) -> Dict[str, torch.Tensor]:
        """Process image and text into model inputs. To be implemented by subclasses.""" 
        pass
        
    @abstractmethod
    def find_text_generation_start(self, input_ids: torch.Tensor) -> int:
        """Find the token position where text generation starts. To be implemented by subclasses."""
        pass
        
    def _setup_device(self, device: str) -> str:
        """Setup compute device."""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
        
    def generate_text(self, image: Image.Image, prompt: str, max_new_tokens: int = 50) -> str:
        """
        Generate text response for image and prompt.
        
        Args:
            image: Input image
            prompt: Text prompt  
            max_new_tokens: Maximum tokens to generate
            
        Returns:
            Generated text response
        """
        inputs = self.process_inputs(image, prompt)
        
        # Apply steering if configured
        if self.steering_vectors:
            return self._generate_with_steering(inputs, max_new_tokens)
        else:
            return self._generate_without_steering(inputs, max_new_tokens)
            
    def _generate_without_steering(self, inputs: Dict[str, torch.Tensor], max_new_tokens: int) -> str:
        """Generate text without activation steering."""
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
        
    def _generate_with_steering(self, inputs: Dict[str, torch.Tensor], max_new_tokens: int) -> str:
        """Generate text with activation steering applied."""
        # This is complex and will be implemented in subclasses
        # For now, fall back to non-steering generation
        return self._generate_without_steering(inputs, max_new_tokens)
        
    def get_activations(self, image: Image.Image, text: str, 
                       layer_names: Optional[List[str]] = None) -> TraceDict:
        """
        Extract activations from specified layers.
        
        Args:
            image: Input image
            text: Input text
            layer_names: List of layer names to extract from (None = all LM layers)
            
        Returns:
            TraceDict containing extracted activations
        """
        if layer_names is None:
            layer_names = self.language_model_layers
            
        inputs = self.process_inputs(image, text)
        
        from utils.nethook import trace_activations
        
        def forward_pass(model, **kwargs):
            return model(**inputs)
            
        return trace_activations(self.model, layer_names, forward_pass)
        
    def collect_activations_batch(self, image_text_pairs: List[Tuple[Image.Image, str]], 
                                 layer_names: Optional[List[str]] = None) -> Dict[str, List[torch.Tensor]]:
        """
        Collect activations from multiple image-text pairs.
        
        Args:
            image_text_pairs: List of (image, text) tuples
            layer_names: Layer names to collect from
            
        Returns:
            Dictionary mapping layer names to lists of activations
        """
        if layer_names is None:
            layer_names = self.language_model_layers
            
        if self.activation_collector is None:
            self.activation_collector = ActivationCollector(self.model, layer_names)
        else:
            self.activation_collector.clear()
            
        for image, text in image_text_pairs:
            inputs = self.process_inputs(image, text)
            
            def forward_pass(model, **kwargs):
                return model(**inputs)
                
            self.activation_collector.collect(forward_pass)
            
        return self.activation_collector.get_all()
        
    def compute_contrastive_vectors(self, positive_pairs: List[Tuple[Image.Image, str]], 
                                   negative_pairs: List[Tuple[Image.Image, str]],
                                   layer_names: Optional[List[str]] = None,
                                   position_index: int = -2) -> Dict[str, torch.Tensor]:
        """
        Compute contrastive activation vectors from positive and negative examples.
        
        Args:
            positive_pairs: List of (image, text) pairs for positive behavior
            negative_pairs: List of (image, text) pairs for negative behavior  
            layer_names: Layer names to compute vectors for
            position_index: Token position to extract activations from (-2 = second-to-last)
            
        Returns:
            Dictionary mapping layer names to steering vectors
        """
        if layer_names is None:
            layer_names = self.language_model_layers
            
        # Collect positive activations
        pos_activations = self.collect_activations_batch(positive_pairs, layer_names)
        
        # Collect negative activations  
        neg_activations = self.collect_activations_batch(negative_pairs, layer_names)
        
        vectors = {}
        for layer_name in layer_names:
            if layer_name in pos_activations and layer_name in neg_activations:
                # Stack and take mean
                pos_stack = torch.stack(pos_activations[layer_name])  # [n_samples, batch, seq, hidden]
                neg_stack = torch.stack(neg_activations[layer_name])
                
                # Extract activations from specific position
                pos_at_position = pos_stack[:, 0, position_index, :]  # [n_samples, hidden]
                neg_at_position = neg_stack[:, 0, position_index, :]
                
                # Compute contrastive vector
                pos_mean = pos_at_position.mean(dim=0)  # [hidden]
                neg_mean = neg_at_position.mean(dim=0)  # [hidden]
                
                vectors[layer_name] = pos_mean - neg_mean
                
        return vectors
        
    def set_steering_vector(self, layer_name: str, vector: torch.Tensor, 
                           multiplier: float = 1.0, from_position: Optional[int] = None) -> None:
        """
        Set a steering vector for a specific layer.
        
        Args:
            layer_name: Name of layer to steer
            vector: Steering vector to apply
            multiplier: Scaling factor for the vector
            from_position: Token position to start applying steering (None = auto-detect)
        """
        self.steering_vectors[layer_name] = vector.to(self.device)
        self.steering_multipliers[layer_name] = multiplier
        self.steering_positions[layer_name] = from_position
        
    def clear_steering(self) -> None:
        """Clear all configured steering vectors."""
        self.steering_vectors.clear()
        self.steering_multipliers.clear() 
        self.steering_positions.clear()
        
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the model architecture."""
        return {
            'model_name': self.model_name,
            'device': self.device,
            'language_model_layers': self.language_model_layers,
            'vision_encoder_layers': self.vision_encoder_layers,
            'num_steering_vectors': len(self.steering_vectors)
        }
        
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(model_name='{self.model_name}', device='{self.device}')"
