# MLLM Task-Vector Dataset

This dataset was generated according to the shapes-spec-v1.0 specification.

## Generation Details

- **Generator**: generate_dataset.py
- **Base seed**: 42
- **Split sizes**: {'A': 500, 'B': 500, 'C': 500}
- **Canvas size**: 256x256 pixels
- **Colors**: ['red', 'green', 'blue', 'yellow']
- **Shapes**: ['circle', 'square', 'triangle']

## Reproducing the Dataset

To regenerate this exact dataset:

```bash
python generate_dataset.py --base_seed 42 --output_dir data
```

## Dataset Structure

```
data/
├── images/
│   ├── A_factor_isolated/     # Factor-isolated scenes (same color & shape)
│   ├── B_mixed/              # Mixed-attribute scenes  
│   └── C_stress/             # Stress/OOD scenes
├── manifests/
│   ├── images.jsonl          # Image metadata (all splits)
│   ├── questions.jsonl       # Questions (mainly Split A matched triples)
│   └── stats.json           # Statistics and validation results
└── README.md                # This file
```

## Usage

The dataset is designed for learning task vectors for:
- **Count**: How many objects are there? (answer: single digit)
- **Color**: What color are the objects? (answer: red/green/blue/yellow)  
- **Shape**: What shape are the objects? (answer: circle/square/triangle)

Split A provides matched question triples for the same image to compute task means independent of content.
