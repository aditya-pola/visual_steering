"""
Activation hooking utilities for multimodal language models.
Based on the concept from the baukit library but simplified for our use case.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Callable, Any
from contextlib import contextmanager
from collections import defaultdict


class TraceDict:
    """
    Dictionary-like object for storing traced activations.
    """
    def __init__(self):
        self.data = {}
        
    def __getitem__(self, key):
        return self.data[key]
        
    def __setitem__(self, key, value):
        self.data[key] = value
        
    def __contains__(self, key):
        return key in self.data
        
    def keys(self):
        return self.data.keys()
        
    def values(self):
        return self.data.values()
        
    def items(self):
        return self.data.items()


class Trace:
    """
    Context manager for tracing intermediate activations in a model.
    """
    def __init__(self, model: nn.Module, layer_names: List[str], retain_output: bool = True):
        self.model = model
        self.layer_names = layer_names if isinstance(layer_names, list) else [layer_names]
        self.retain_output = retain_output
        self.trace_dict = TraceDict()
        self.hooks = []
        
    def __enter__(self):
        # Register hooks for each specified layer
        for name in self.layer_names:
            layer = self._get_layer_by_name(name)
            if layer is not None:
                hook = layer.register_forward_hook(
                    self._make_hook(name)
                )
                self.hooks.append(hook)
        return self.trace_dict
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        # Remove all hooks
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def _get_layer_by_name(self, name: str) -> Optional[nn.Module]:
        """Get a layer by its name path (e.g., 'model.layers.0')."""
        parts = name.split('.')
        obj = self.model
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif part.isdigit() and hasattr(obj, '__getitem__'):
                obj = obj[int(part)]
            else:
                return None
        return obj
        
    def _make_hook(self, name: str):
        """Create a hook function for a given layer name."""
        def hook_fn(module, input, output):
            if self.retain_output:
                # Store the output activation
                if isinstance(output, torch.Tensor):
                    self.trace_dict[name] = output.detach()
                elif isinstance(output, (tuple, list)) and len(output) > 0:
                    # For transformer blocks that return (hidden_states, ...) 
                    self.trace_dict[name] = output[0].detach()
                else:
                    self.trace_dict[name] = output
        return hook_fn


class TraceWithEdit:
    """
    Context manager for both tracing and editing activations.
    """
    def __init__(self, model: nn.Module, layer_names: List[str], edit_func: Callable = None):
        self.model = model
        self.layer_names = layer_names if isinstance(layer_names, list) else [layer_names]
        self.edit_func = edit_func
        self.trace_dict = TraceDict()
        self.hooks = []
        
    def __enter__(self):
        for name in self.layer_names:
            layer = self._get_layer_by_name(name)
            if layer is not None:
                hook = layer.register_forward_hook(
                    self._make_hook(name)
                )
                self.hooks.append(hook)
        return self.trace_dict
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        
    def _get_layer_by_name(self, name: str) -> Optional[nn.Module]:
        """Get a layer by its name path."""
        parts = name.split('.')
        obj = self.model
        for part in parts:
            if hasattr(obj, part):
                obj = getattr(obj, part)
            elif part.isdigit() and hasattr(obj, '__getitem__'):
                obj = obj[int(part)]
            else:
                return None
        return obj
        
    def _make_hook(self, name: str):
        """Create a hook that traces and optionally edits."""
        def hook_fn(module, input, output):
            # Store original output
            if isinstance(output, torch.Tensor):
                original = output.detach()
                self.trace_dict[name] = original
            elif isinstance(output, (tuple, list)) and len(output) > 0:
                original = output[0].detach()
                self.trace_dict[name] = original
            else:
                self.trace_dict[name] = output
                return output
                
            # Apply edit function if provided
            if self.edit_func is not None:
                if isinstance(output, torch.Tensor):
                    edited = self.edit_func(output, name)
                    return edited
                elif isinstance(output, (tuple, list)):
                    edited_first = self.edit_func(output[0], name)
                    return (edited_first,) + output[1:]
                    
            return output
        return hook_fn


def get_layer_names(model: nn.Module, layer_type: str = 'transformer') -> List[str]:
    """
    Get standard layer names for a model type.
    
    Args:
        model: The model to get layer names for
        layer_type: Type of layers ('transformer', 'attention', 'mlp', etc.)
    
    Returns:
        List of layer name strings
    """
    layer_names = []
    
    # Try to detect model architecture
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        # GPT-style models
        for i in range(len(model.transformer.h)):
            layer_names.append(f'transformer.h.{i}')
            
    elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
        # Llama/Gemma-style models  
        for i in range(len(model.model.layers)):
            layer_names.append(f'model.layers.{i}')
            
    elif hasattr(model, 'language_model') and hasattr(model.language_model.model, 'layers'):
        # Multimodal models with separate language model
        for i in range(len(model.language_model.model.layers)):
            layer_names.append(f'language_model.model.layers.{i}')
            
    elif hasattr(model, 'layers'):
        # Simple case
        for i in range(len(model.layers)):
            layer_names.append(f'layers.{i}')
            
    return layer_names


def trace_activations(model: nn.Module, layer_names: List[str], input_func: Callable, **kwargs) -> TraceDict:
    """
    Convenience function to trace activations for a single forward pass.
    
    Args:
        model: The model to trace
        layer_names: Names of layers to trace
        input_func: Function that performs the forward pass when called with model
        **kwargs: Additional arguments passed to input_func
    
    Returns:
        TraceDict containing traced activations
    """
    with Trace(model, layer_names) as trace:
        input_func(model, **kwargs)
    return trace


def edit_activations(model: nn.Module, layer_names: List[str], edit_func: Callable, 
                    input_func: Callable, **kwargs) -> tuple:
    """
    Convenience function to edit and trace activations for a single forward pass.
    
    Args:
        model: The model to trace and edit
        layer_names: Names of layers to edit
        edit_func: Function to apply to activations (takes activation tensor and layer name)
        input_func: Function that performs the forward pass
        **kwargs: Additional arguments passed to input_func
    
    Returns:
        Tuple of (model_output, trace_dict)
    """
    with TraceWithEdit(model, layer_names, edit_func) as trace:
        output = input_func(model, **kwargs)
    return output, trace


class ActivationCollector:
    """
    Utility class for collecting activations across multiple forward passes.
    """
    def __init__(self, model: nn.Module, layer_names: List[str]):
        self.model = model
        self.layer_names = layer_names
        self.activations = defaultdict(list)
        
    def collect(self, input_func: Callable, **kwargs) -> None:
        """Collect activations from a single forward pass."""
        with Trace(self.model, self.layer_names) as trace:
            input_func(self.model, **kwargs)
            
        # Store activations
        for layer_name in self.layer_names:
            if layer_name in trace:
                self.activations[layer_name].append(trace[layer_name])
                
    def get_means(self) -> Dict[str, torch.Tensor]:
        """Get mean activations across all collected samples."""
        means = {}
        for layer_name in self.layer_names:
            if layer_name in self.activations and self.activations[layer_name]:
                stacked = torch.stack(self.activations[layer_name])
                means[layer_name] = stacked.mean(dim=0)
        return means
        
    def get_all(self) -> Dict[str, List[torch.Tensor]]:
        """Get all collected activations."""
        return dict(self.activations)
        
    def clear(self) -> None:
        """Clear all collected activations."""
        self.activations.clear()


def add_vector_to_activations(activation: torch.Tensor, vector: torch.Tensor, 
                             position_ids: Optional[torch.Tensor] = None,
                             from_position: Optional[int] = None) -> torch.Tensor:
    """
    Add a steering vector to activations, optionally from a specific position onwards.
    
    Args:
        activation: Original activation tensor [batch_size, seq_len, hidden_dim]
        vector: Steering vector to add [hidden_dim]
        position_ids: Position IDs for each token [batch_size, seq_len]  
        from_position: Position index to start adding from (None = add to all positions)
        
    Returns:
        Modified activation tensor
    """
    # Ensure vector is on same device and dtype as activation
    vector = vector.to(device=activation.device, dtype=activation.dtype)
    
    if from_position is None:
        # Add to all positions
        expanded_vector = vector.unsqueeze(0).unsqueeze(0).to(activation.dtype)
        return activation + expanded_vector
    
    if position_ids is None:
        # Use simple indexing
        modified = activation.clone()
        expanded_vector = vector.unsqueeze(0).unsqueeze(0).to(activation.dtype)
        if from_position >= 0:
            # Positive indexing
            if from_position < activation.shape[1]:
                modified[:, from_position:, :] += expanded_vector
        else:
            # Negative indexing from end
            end_pos = activation.shape[1] + from_position
            if end_pos >= 0:
                modified[:, end_pos:, :] += expanded_vector
        return modified
    else:
        # Use position IDs for masking
        position_ids = position_ids.to(activation.device)
        mask = (position_ids >= from_position).unsqueeze(-1).to(activation.dtype)
        expanded_vector = vector.unsqueeze(0).unsqueeze(0).to(activation.dtype)
        return activation + mask * expanded_vector


@contextmanager
def no_grad():
    """Context manager that disables gradient computation."""
    with torch.no_grad():
        yield
