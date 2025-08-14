# Contrastive Activation Addition (CAA) for Multimodal LLMs: Analysis and Implementation Plan

## 1. Analysis of Existing CAA Implementation

### 1.1 Core CAA Methodology

Based on the analysis of the CAA codebase, Contrastive Activation Addition works through the following key steps:

1. **Contrastive Dataset Creation**: 
   - Pairs of prompts with positive and negative behavioral responses
   - Format: `{"question": "...", "answer_matching_behavior": "A", "answer_not_matching_behavior": "B"}`
   - Examples include sycophancy, refusal, hallucination, etc.

2. **Activation Extraction**:
   - Forward pass through model with positive and negative examples
   - Extract activations from specific layers (typically residual stream after attention+MLP)
   - Store activations from the second-to-last token position (`-2` index)

3. **Steering Vector Computation**:
   - For each layer: `steering_vector = mean(positive_activations) - mean(negative_activations)`
   - Vectors capture the direction in activation space that corresponds to the behavioral difference

4. **Steering Application**:
   - During inference, add `multiplier * steering_vector` to activations at specific layers
   - Applied from a specific token position onwards (typically after instruction end)

### 1.2 Key Components in LlamaWrapper

The CAA implementation uses several crucial components:

- **BlockOutputWrapper**: Wraps each transformer block to intercept and modify activations
- **AttnWrapper**: Wraps attention mechanism to save intermediate activations  
- **Activation Injection**: Uses `add_vector_from_position()` to modify activations during forward pass
- **Position Detection**: Finds instruction end position to determine where to start applying steering

### 1.3 CAA Architecture Pattern

```python
# Simplified CAA flow:
1. Wrap model layers with BlockOutputWrapper
2. During generation of positive/negative examples:
   - Forward pass → Extract activations at layer L, position -2
   - Store positive_acts[L] and negative_acts[L]
3. Compute steering vector: vec[L] = mean(pos_acts[L]) - mean(neg_acts[L])  
4. During inference with steering:
   - Add multiplier * vec[L] to activations at layer L from position P onwards
```

## 2. Challenges for Multimodal LLMs

### 2.1 Architectural Differences

**PaliGemma Architecture**:
- Vision Encoder (SigLIP) → Image embeddings
- Projection layer → Align vision and text dimensions  
- Language Model (Gemma) → Generate text
- **Key insight**: We want to steer only the language model component

**Challenges**:
1. **Multi-input handling**: Both image and text tokens flow through the language model
2. **Token position mapping**: Need to identify text-only tokens vs. image tokens
3. **Attention patterns**: Vision-language attention may complicate steering
4. **Different tokenization**: Vision tokens vs. text tokens have different semantics

### 2.2 Activation Extraction Complexity

- **PaliGemma**: Image tokens are prepended, text tokens follow
- **Position detection**: Need to find where text generation starts (not just instruction end)
- **Multi-modal attention**: Image tokens influence text generation throughout

## 3. Implementation Plan for MLLM CAA

### 3.1 High-Level Architecture

```
Dataset (Visual Q&A) → Activation Extraction → Vector Computation → Steering Application
        ↓                        ↓                      ↓                ↓
    Images + Text         Extract from LM only     Task vectors     Steer text generation
```

### 3.2 Core Components to Implement

#### 3.2.1 MLLM Wrapper (src/mllm_wrapper.py)

```python
class MLLMWrapper:
    """
    Generic wrapper for multimodal LLMs supporting CAA steering
    Focuses on steering the language model component only
    """
    - Abstract base class for different MLLM architectures
    - Handle vision encoder + language model separation  
    - Activation extraction from language model layers only
    - Position-aware steering (avoid steering on vision tokens)
```

#### 3.2.2 PaliGemma Wrapper (src/paligemma_wrapper.py)

```python
class PaliGemmaWrapper(MLLMWrapper):
    """
    PaliGemma-specific implementation of MLLM wrapper
    """
    - Inherit from MLLMWrapper
    - Handle PaliGemma's specific architecture
    - Extract activations from Gemma language model layers
    - Implement position detection for text generation start
```

#### 3.2.3 Activation Hooks (src/activation_hooks.py)

```python
class MLLMBlockWrapper:
    """
    Modified BlockOutputWrapper for multimodal models
    """
    - Wrap language model transformer blocks
    - Save activations during forward pass
    - Apply steering vectors at appropriate positions
    - Handle multi-modal token masking
```

#### 3.2.4 Task Vector Generation (src/generate_task_vectors.py)

```python
def generate_task_vectors_mllm():
    """
    Generate task vectors using our visual dataset
    """
    - Use our count/color/shape dataset from Split A
    - Extract activations for matched triples (same image, different questions)
    - Compute task vectors: vec_count = mean(count_acts) - mean(neutral_acts)
```

### 3.3 Integration with Existing Dataset

Our generated dataset is perfect for CAA because:

1. **Split A Factor-Isolated**: Provides clean task-specific examples
2. **Matched Triples**: Same image with count/color/shape questions enables controlled comparison
3. **Single-token Responses**: Clean activation extraction from response tokens
4. **Neutral Baseline**: Can use generic "Answer briefly" as neutral prompt

### 3.4 Implementation Steps

#### Phase 1: Basic MLLM Wrapper
1. Create `src/mllm_wrapper.py` with abstract base class
2. Implement `src/paligemma_wrapper.py` extending our evaluation script
3. Add activation extraction capabilities using nethook utilities
4. Test basic forward pass and activation saving

#### Phase 2: Task Vector Generation  
1. Create `src/generate_task_vectors.py` using our dataset
2. Implement position-aware activation extraction
3. Generate count/color/shape task vectors from Split A
4. Validation: verify vectors capture task-specific information

#### Phase 3: Steering Implementation
1. Add steering capability to PaliGemmaWrapper
2. Test steering with different multipliers and layers
3. Evaluate on Split B (mixed attributes) for generalization
4. Compare steered vs. unsteered responses

#### Phase 4: Extension Framework
1. Design generic interface for other HuggingFace VL models
2. Test with LLaVA, InstructBLIP, etc.
3. Create utilities for easy model extension

### 3.5 Key Technical Considerations

#### 3.5.1 Position Detection Strategy

```python
def find_text_generation_start(input_ids, image_token_count):
    """
    Find where text generation starts in multimodal input
    """
    # For PaliGemma: image tokens + text prompt + generation start
    return image_token_count + prompt_length
```

#### 3.5.2 Activation Masking

```python  
def apply_steering_masked(activations, steering_vector, mask):
    """
    Apply steering only to text generation tokens, not image tokens
    """
    text_mask = mask[image_token_count:]  # Mask for text tokens only
    return activations + (text_mask * steering_vector)
```

#### 3.5.3 Task-Specific Vector Computation

```python
# Our dataset provides perfect contrastive pairs:
# Same image, different tasks
count_activations = extract_activations(image, "How many objects?")
color_activations = extract_activations(image, "What color?") 
neutral_activations = extract_activations(image, "Answer briefly.")

count_vector = count_activations - neutral_activations
color_vector = color_activations - neutral_activations  
```

## 4. Expected Outcomes

### 4.1 Task Vector Capabilities

With our implementation, we should be able to:

1. **Generate task vectors** for count, color, shape from our dataset
2. **Steer neutral prompts** to focus on specific tasks:
   - Input: Image + "Answer briefly" → Output: Counts objects
   - Input: Image + "Describe this" → Output: Focuses on colors

3. **Control task priority** in mixed scenarios:
   - Apply count vector → Emphasizes counting over other attributes
   - Apply color vector → Emphasizes colors over other attributes

### 4.2 Research Applications

1. **Task Routing**: Automatically direct model attention to specific visual tasks
2. **Behavioral Analysis**: Understand how multimodal models process different question types  
3. **Robustness Testing**: How do task vectors generalize across different images/contexts?
4. **Model Interpretability**: What visual features do count/color/shape vectors capture?

## 5. File Structure

```
src/
├── __init__.py
├── mllm_wrapper.py          # Abstract base class for MLLM wrappers
├── paligemma_wrapper.py     # PaliGemma-specific implementation  
├── activation_hooks.py      # Modified block wrappers for activation injection
├── generate_task_vectors.py # Generate vectors from our visual dataset
├── apply_steering.py        # Apply task vectors during inference
└── utils/
    ├── __init__.py
    ├── position_utils.py    # Position detection utilities
    ├── activation_utils.py  # Activation extraction utilities
    └── model_utils.py       # Model loading and setup utilities

utils/
└── nethook.py               # Activation hooking utilities (to be populated)
```

## 6. Integration with Existing Codebase

### 6.1 Extend Evaluation Script

Our existing `evaluate_baseline.py` can be extended to:
1. Save activations during evaluation
2. Apply task vectors and measure steering effects  
3. Compare steered vs. baseline performance

### 6.2 New Evaluation Modes

```bash
# Generate task vectors from our dataset
python generate_task_vectors.py --dataset_dir data --model google/paligemma2-3b-mix-224

# Evaluate with task vector steering  
python evaluate_baseline.py --apply_steering --task_vector count --multiplier 1.0

# Test cross-task generalization
python evaluate_baseline.py --apply_steering --task_vector color --splits B,C
```

## 7. Advantages of This Approach

1. **Controlled Dataset**: Our factor-isolated Split A provides clean training data
2. **Single-token Targets**: Easier activation extraction compared to open-ended generation
3. **Matched Comparisons**: Same image with different questions enables precise vector computation
4. **HuggingFace Integration**: Leverages existing model loading and tokenization
5. **Extensible Design**: Framework can be adapted to other vision-language models

This implementation plan provides a systematic approach to bringing CAA to multimodal LLMs while leveraging our existing dataset and evaluation infrastructure.
