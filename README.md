# Visual Steering for Multimodal Large Language Models

This repository implements Contrastive Activation Addition (CAA) for multimodal Large Language Models (MLLMs), specifically adapted for vision-language models like PaliGemma. It includes a complete pipeline from dataset generation to steering evaluation.

## Overview

The project implements visual steering using factor-isolated datasets to learn task-specific activation vectors for guiding MLLM behavior during inference. Unlike text-only steering, this approach works with image+text inputs to influence visual reasoning.

### Key Components

1. **Dataset Generation**: Creates 2D shape images with associated question-answer pairs following controlled splits
2. **MLLM Wrapper Framework**: Extensible architecture for different vision-language models
3. **Task Vector Generation**: Learns contrastive activation vectors from matched question triples
4. **Steering Evaluation**: Comprehensive analysis of steering effects and cross-task interference
5. **Baseline Evaluation**: Traditional model evaluation for comparison

## Quick Start

### Environment Setup

```bash
# Activate the visteer conda environment
conda activate visteer

# Dependencies should already be installed, but if needed:
pip install torch transformers tqdm accelerate matplotlib numpy pillow
```

### Test Implementation

```bash
# Validate the full visual steering implementation
python run_test.py
```

### Full Visual Steering Pipeline

```bash
# Run complete pipeline: dataset → vectors → evaluation
python scripts/run_pipeline.py --max_images 100 --max_questions 50
```

### Individual Components

#### 1. Generate Dataset
```bash
# Generate full dataset (500 images per split)
python scripts/generate_dataset.py

# Generate smaller dataset for testing
python scripts/generate_dataset.py --split_a_size 100 --split_b_size 100 --split_c_size 100
```

#### 2. Generate Task Vectors
```bash
# Extract steering vectors from dataset
python src/generate_task_vectors.py \
    --model google/paligemma2-3b-mix-224 \
    --dataset_dir data \
    --output_dir task_vectors \
    --max_images 200
```

#### 3. Evaluate Steering
```bash
# Compare baseline vs steered performance
python src/evaluate_steering.py \
    --model google/paligemma2-3b-mix-224 \
    --dataset_dir data \
    --task_vectors_dir task_vectors \
    --eval_baseline \
    --eval_cross_task
```

#### 4. Baseline Evaluation (Traditional)
```bash
# Standard model evaluation
python scripts/evaluate_baseline.py --quick
```

## Architecture

### Visual Steering Framework

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Dataset       │    │  Task Vectors   │    │   Steering      │
│   Generation    │───▶│   Generation    │───▶│   Evaluation    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
  Factor-isolated         Contrastive            Baseline vs
  visual questions        activations             steered comparison
```

### Core Components

1. **MLLM Wrapper Framework** (`src/mllm_wrapper.py`)
   - Abstract base class for multimodal model integration
   - Standardized activation extraction and steering interfaces
   - Extensible to other HuggingFace vision-language models

2. **PaliGemma Implementation** (`src/paligemma_wrapper.py`)
   - Model-specific wrapper with device management
   - Block-level activation steering using custom wrappers
   - Handles float16/float32 dtype compatibility

3. **Task Vector Generation** (`src/generate_task_vectors.py`) 
   - Extracts contrastive activations from matched question triples
   - Computes steering vectors: task_mean - neutral_mean
   - Supports multiple layers and position-aware steering

4. **Activation Hooks** (`utils/nethook.py`)
   - Low-level utilities for tracing and editing activations
   - Position-aware vector addition during forward pass
   - Handles multi-GPU device distribution

### Device Management

The implementation provides robust device handling for both single and multi-GPU setups:

```python
from src.paligemma_wrapper import PaliGemmaWrapper, get_cuda0_device_config

# Force all model components to CUDA:0 (avoids multi-GPU issues)
device = get_cuda0_device_config()
model = PaliGemmaWrapper("google/paligemma2-3b-mix-224", device=device)

# Or use automatic device mapping
model = PaliGemmaWrapper("google/paligemma2-3b-mix-224", device="auto")
```

## Dataset Structure

```
data/
├── images/
│   ├── A_factor_isolated/    # Same color & shape (for task vectors)
│   ├── B_mixed/              # Mixed attributes (for evaluation)  
│   └── C_stress/             # Stress conditions (robustness)
├── manifests/
│   ├── images.jsonl          # Image metadata
│   ├── questions.jsonl       # Question-answer pairs
│   └── stats.json            # Dataset statistics
└── README.md
```

### Task Vector Pipeline Structure

```
visual_steering/
├── scripts/                     # Main executable scripts
│   ├── generate_dataset.py      # Dataset generation
│   ├── evaluate_baseline.py     # Traditional evaluation
│   ├── test_implementation.py   # Implementation validation
│   ├── run_pipeline.py         # Full pipeline orchestration
│   ├── run_evaluations.py      # Evaluation runner
│   └── eval.sh                 # Shell script for evaluations
├── src/                        # Core implementation modules
│   ├── mllm_wrapper.py         # Abstract base class
│   ├── paligemma_wrapper.py    # PaliGemma implementation
│   ├── generate_task_vectors.py # Vector generation
│   └── evaluate_steering.py    # Steering evaluation
├── utils/
│   └── nethook.py              # Activation hooks
├── data/                       # Generated dataset
├── task_vectors/               # Generated steering vectors
├── experiments/                # Evaluation results
├── run_test.py                 # Test runner wrapper
└── requirements.txt           # Dependencies
```

## Results

### Implementation Validation

The visual steering implementation successfully demonstrates:

- ✅ **Model Loading**: PaliGemma loads correctly with single-GPU device mapping
- ✅ **Activation Extraction**: Extracts 2304-dimensional vectors from 26 language model layers  
- ✅ **Steering Application**: Applies steering vectors during generation without dtype issues
- ✅ **Dataset Integration**: Processes 1500 factor-isolated questions from Split A
- ✅ **Task Vector Generation**: Creates contrastive vectors for count, color, and shape tasks
- ✅ **Cross-Task Analysis**: Framework ready for evaluating steering effects

### Baseline Performance (Split A - Factor Isolated)

**PaliGemma-2-3B-Mix-224 Performance:**

| Task  | Accuracy | Sample Size | Notes |
|-------|----------|-------------|-------|
| Color | 100.0%   | 500 questions | Perfect color recognition |  
| Count | 85.2%    | 500 questions | Some difficulty with higher counts |
| Shape | 98.8%    | 500 questions | Near-perfect shape recognition |
| **Overall** | **94.7%** | **1500 questions** | Strong baseline for steering comparison |

### Technical Achievements

1. **Multi-GPU Compatibility**: Handles PaliGemma's distributed architecture
2. **Dtype Safety**: Seamless float16/float32 conversion in activation paths
3. **Position-Aware Steering**: Applies vectors at specific token positions during generation
4. **Block-Level Integration**: Minimal modification to original model architecture
5. **Extensible Design**: Framework supports adding new vision-language models

## Key Features

### Dataset Splits for Steering

- **Split A (Factor-Isolated)**: All objects in each image share the same color and shape
  - Critical for computing clean task vectors
  - Provides matched question triples (count/color/shape) for the same image
  - 500 images with 1500 total questions (500 per task)
  
- **Split B (Mixed-Attribute)**: Images contain multiple colors and shapes
  - Tests steering generalization and routing capabilities
  - Evaluates interference between different visual attributes
  
- **Split C (Stress/OOD)**: Challenging conditions for robustness testing
  - Occlusions, low contrast, color confusion, extreme sizes
  - Tests steering vector robustness under visual stress

### Steering Strategy

1. **Contrastive Vectors**: `task_activations - neutral_activations`
2. **Layer Targeting**: 26 language model layers in PaliGemma
3. **Position-Aware**: Applies steering to specific token positions
4. **Scale Control**: Configurable steering intensity (0.1x - 2.0x)

### Task Types

All tasks have constrained single-token outputs for clean evaluation:

- **Count**: "How many objects?" → Single digit (0-9)
- **Color**: "What color?" → {red, green, blue, yellow} 
- **Shape**: "What shape?" → {circle, square, triangle}

### Evaluation Framework

The steering evaluation provides:
- Baseline vs steered performance comparison
- Cross-task interference analysis
- Layer-wise steering effectiveness
- Statistical significance testing
- Detailed failure case analysis

## Model Support & Extension

### Supported Models

The framework currently supports:
- **PaliGemma models** (fully implemented and tested)
- **Extensible to other HuggingFace VL models** following similar patterns

### Adding New Models

To extend support to other vision-language models:

```python
class YourModelWrapper(MLLMWrapper):
    def load_model(self):
        # Model-specific loading logic
        self.model = AutoModelForVision2Seq.from_pretrained(self.model_name)
        
    def get_language_model_layers(self):
        # Return steerable layer names
        return ["transformer.layers.0", "transformer.layers.1", ...]
        
    def process_inputs(self, image, text):
        # Convert to model format
        return self.processor(images=image, text=text, return_tensors="pt")
```

The abstract `MLLMWrapper` class handles:
- Activation extraction interface
- Steering vector management  
- Position-aware vector application
- Device and dtype compatibility

### Device Configuration Functions

```python
# Force single GPU (recommended for consistency)
device = get_cuda0_device_config()
model = PaliGemmaWrapper(model_name, device=device)

# Alternative single GPU configuration  
device = get_single_gpu_device_config()
model = YourModelWrapper(model_name, device=device)
```

## Implementation Details

### Activation Steering Process

1. **Vector Extraction**: Process matched question triples from Split A
2. **Contrastive Computation**: `task_mean - neutral_mean` across layer activations
3. **Block Wrapping**: Inject custom wrappers into language model layers
4. **Forward Hook**: Apply steering vectors during generation
5. **Position Targeting**: Modify activations at specific token positions

### Technical Innovations

- **Multi-GPU Device Mapping**: Handles distributed model architectures
- **Dtype Safety**: Automatic float16/float32 conversion in activation paths  
- **Minimal Model Modification**: Uses forward hooks instead of model surgery
- **Position-Aware Steering**: Applies vectors only during text generation phase
- **Extensible Architecture**: Abstract base class supports multiple model types

### Performance Considerations

- **Memory Efficient**: Streaming activation collection for large datasets
- **GPU Optimized**: Leverages native model distribution and compilation
- **Batch Processing**: Supports batch evaluation for faster analysis
- **Configurable Precision**: Balances speed vs accuracy with dtype selection

## Next Steps & Future Work

This implementation provides the foundation for advanced visual steering research:

### Immediate Applications
1. **Task Vector Quality Analysis**: Compare vector effectiveness across layers
2. **Steering Scale Optimization**: Find optimal intensity for each task
3. **Generalization Studies**: Test vectors on Split B/C for robustness
4. **Model Comparison**: Evaluate steering across different VL architectures

### Research Extensions  
1. **Multi-Task Vectors**: Learn vectors for combined visual reasoning tasks
2. **Compositional Steering**: Apply multiple vectors simultaneously
3. **Adversarial Robustness**: Test steering against adversarial visual inputs
4. **Real-World Evaluation**: Apply to natural images beyond synthetic shapes

### Technical Improvements
1. **Automatic Layer Selection**: Learn which layers are most steerable
2. **Dynamic Position Targeting**: Adaptive position selection during generation
3. **Vector Interpolation**: Smooth transitions between different steering directions
4. **Efficiency Optimizations**: Faster vector extraction and application
