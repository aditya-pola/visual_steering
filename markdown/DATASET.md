# MLLM Task-Vector Dataset Spec (Matplotlib 2D Shapes)

This spec defines **exact instructions** for generating a dataset of 2D shape images (using matplotlib) and the **JSONL manifests** that describe them. The dataset is designed to learn **single-dimension task vectors** for **count**, **color**, and **shape** that can be applied at test time without restating the task.

---

## 0) High-level goals

- Provide **factor-isolated** scenes for building *task* vectors (not content vectors).
- Provide **mixed** scenes for evaluating robustness and routing on varied content.
- Provide an optional **stress** split for occlusions, low contrast, and color confusion.
- Provide **per-image matched triples** of prompts (count/color/shape) for the same scene (used to compute task means independent of content).

---

## 1) Canonical rendering setup (matplotlib)

- **Canvas**: white background, size **256×256 px** (square).  
  - Matplotlib: `plt.figure(figsize=(2.56, 2.56), dpi=100)` gives 256 px; or render to 512 px if you prefer and downsample.
  - Turn off axes: `ax.axis('off')` and set `ax.set_xlim(0, 256); ax.set_ylim(0, 256); ax.set_aspect('equal')`.
  - Save as PNG (no alpha): `bbox_inches='tight', pad_inches=0`.
- **Coordinate system**: pixel coordinates in **[0, 256] × [0, 256]** with origin at **bottom-left**. (If you keep matplotlib’s top-left origin, reflect Y when writing JSON.)
- **Shapes**: `circle`, `square`, `triangle`.
- **Colors** (token-friendly): `red`, `green`, `blue`, `yellow` with RGB hex:
  - red: `#FF3B30`
  - green: `#34C759`
  - blue: `#007AFF`
  - yellow: `#FFCC00`
- **Sizes** (in pixels, inclusive ranges; uniformly sampled):
  - circle **radius**: 14–28
  - square **side**: 24–44
  - triangle **side** (equilateral): 26–48
- **Margins & separation**:
  - Keep **6 px** min margin to canvas edges.
  - Enforce **non-overlap** in the **factor-isolated split**: center-to-center constraints using *effective radii* (see below).
  - Mixed/stress splits may allow controlled overlaps; specify exact policy in JSON.
- **Drawing order**: randomize z-order; record it in JSON as `z` for reproducibility.
- **Randomness**: set a per-image **seed** (integer) and log it in JSON. All stochastic choices (positions, sizes, rotations) must derive from this seed.

**Effective radius** (for quick overlap checks):
- circle: `r = radius`
- square: `r = side / √2` (circumscribed circle)
- triangle: `r = side / √3` (circumscribed circle for equilateral)

Two objects **A,B** do not overlap if `dist(centerA, centerB) >= rA + rB + sep_buffer`, where `sep_buffer = 2 px`.

**Triangle rotation**: random θ ∈ [0°, 360°). Store `rotation_deg` in JSON.

---

## 2) Splits and balancing

Create three splits; the agent must follow **counts and balance** strictly.

### Split A — *Factor-Isolated* (for vector estimation)
- **Purpose**: build task vectors; remove confounds.
- **Scene rule**: All objects in an image share the **same color** and **same shape**.
- **Count**: uniformly sample objects per image from **{1,2,3,4,5,6}**.
- **Non-overlap**: **required** (use rejection sampling/Poisson-disk-like placement).
- **Recommended size**: **500 images** minimum.
- **Balancing across dataset**:
  - Equal frequency of **colors** (R/G/B/Y).
  - Equal frequency of **shapes** (circle/square/triangle).
  - Uniform **count** distribution.

### Split B — *Mixed-Attribute* (for evaluation & routing)
- **Purpose**: evaluate generalization & routing on varied scenes.
- **Scene rule**: An image may include **multiple colors** and **multiple shapes**.
- **Count**: 1–6 total objects per image (uniform).
- **Overlap**: **not allowed** (same as Split A) for the default variant.
- **Balancing**:
  - Approx. uniform marginal distributions of colors and shapes.
  - Balanced co-occurrences (avoid strong color–shape correlations).
- **Recommended size**: **500 images** minimum.

### Split C — *Stress/OOD* (optional but recommended)
- **Purpose**: robustness under realistic noise.
- **Variants** (tag by `stress_type` in JSON; an image can have ≥1 tags):
  - `occlusion_30` or `occlusion_50`: allow overlaps targeting ~30% or ~50% total occluded area.
  - `low_contrast`: reduce color saturation/brightness 20–40% while staying recognizable.
  - `color_confusion`: use **green vs yellow** in adjacent luminance (hard pair).
  - `size_extreme`: push sizes to edges of ranges; include one very small object.
- **Recommended size**: **500 images**.

> **Note:** Split C is never used to compute task means. It is only for evaluation.

---

## 3) Per-image matched *task prompts* (for Split A only)

For **every image in Split A**, create **three QA entries** (same image, different task prompt). Use these **exact templates** to enforce single-token outputs:

- **Count (digits)**  
  `Answer with one digit (0–9) only. How many objects are there?`

- **Color (one word from set)**  
  `Answer with one word only from {red, green, blue, yellow}. What color are the objects?`

- **Shape (one word from set)**  
  `Answer with one word only from {circle, square, triangle}. What shape are the objects?`

Store these in `questions.jsonl` (see schema below). Prompt text must be **identical** across the dataset (no paraphrases) unless you introduce a controlled `prompt_template_id` and keep it perfectly balanced across tasks.

---

## 4) Folder structure

```
dataset_root/
├── images/
│   ├── A_factor_isolated/
│   │   ├── img_A_000001.png
│   │   └── ...
│   ├── B_mixed/
│   │   ├── img_B_000001.png
│   │   └── ...
│   └── C_stress/
│       ├── img_C_000001.png
│       └── ...
├── manifests/
│   ├── images.jsonl            # one line per image (all splits)
│   ├── questions.jsonl         # matched triples for Split A (+ optional eval Qs for B/C)
│   └── stats.json              # global counts & sanity metrics
└── README.md                   # generator/version info
```

---

## 5) `images.jsonl` schema (one JSON object per image)

Each line is a UTF-8 JSON object with fields below. Required unless marked optional.

```json
{
  "image_id": "img_A_000123",
  "split": "A",                          // "A" | "B" | "C"
  "seed": 123456789,
  "canvas": { "width": 256, "height": 256, "bg_color": "#FFFFFF" },
  "stress": {                            // only for split C; else empty object {}
    "occlusion_target": 0.0,            // 0.0, 0.3, 0.5
    "low_contrast": false,
    "color_confusion": false,
    "size_extreme": false
  },
  "objects": [
    {
      "id": "o1",
      "shape": "circle",                 // "circle" | "square" | "triangle"
      "color": "green",                  // "red" | "green" | "blue" | "yellow"
      "center": { "x": 64.0, "y": 192.0 },
      "size": {                          // ONE of these populated depending on shape
        "radius": 22.0,                  // circle
        "side": null                     // square/triangle
      },
      "rotation_deg": 315.0,             // triangles > 0; else 0 for circle/square
      "z": 2,                            // draw order; 0 = background-first
      "bbox_xywh": [42.0, 170.0, 44.0, 44.0],
      "area_px": 1520.5
    }
    // ... more objects
  ],
  "scene_constraints": {
    "min_edge_margin_px": 6.0,
    "non_overlap": true,                 // Split A & B true; Split C may be false
    "sep_buffer_px": 2.0,
    "effective_radius_formula": "see spec v1.0"
  },
  "derived_labels": {
    "count_total": 4,
    "counts_by_color": { "red": 0, "green": 3, "blue": 0, "yellow": 1 },
    "counts_by_shape": { "circle": 1, "square": 2, "triangle": 1 },
    "counts_by_color_shape": {
      "green": { "circle": 1, "square": 2, "triangle": 0 },
      "yellow": { "circle": 0, "square": 0, "triangle": 1 }
    },
    "dominant_color_rule": "pixel_area",
    "dominant_color": "green",
    "majority_shape_rule": "count_then_area_then_leftmost",
    "majority_shape": "square"
  },
  "render": {
    "matplotlib": {
      "backend": "Agg",
      "dpi": 100,
      "antialias": true
    },
    "file": "images/A_factor_isolated/img_A_000123.png"
  },
  "generator_version": "shapes-spec-v1.0"
}
```

**Notes**
- `bbox_xywh` is the axis-aligned bounding box in pixel coordinates `[x_min, y_min, width, height]` (origin bottom-left). If your code uses top-left origin, convert consistently.
- `area_px`: polygonal area in pixels; used to compute `dominant_color` and to break ties for `majority_shape`.
- For **Split A**, all objects must have **same color** and **same shape** (enforce, then assert in a validator).

---

## 6) `questions.jsonl` schema

### 6.1 Required matched triples for **Split A**

One JSON object per **(image_id, task)** pair; three lines per image in Split A.

```json
{
  "image_id": "img_A_000123",
  "split": "A",
  "task": "count",                       // "count" | "color" | "shape"
  "prompt_template_id": "count_v1",
  "prompt_text": "Answer with one digit (0–9) only. How many objects are there?",
  "answer_space": ["0","1","2","3","4","5","6","7","8","9"],
  "ground_truth": "4",
  "answer_rule": "exact_digit",
  "answer_bottleneck_token_index": 0     // first token of the model’s answer
}
```

Color and shape entries differ only in `task`, `prompt_template_id`, `prompt_text`, `answer_space`, and `ground_truth` (derived from `images.jsonl`: dominant color for Split B/C if you also include evaluation prompts there; for Split A, color/shape are trivial).

### 6.2 Optional evaluation QAs for **Split B/C**

You may add QA entries to evaluate on mixed/stress scenes with **explicit tasks** (not needed for computing vectors, but useful for accuracy and routing):

- **Count**: same as above (ground truth is `derived_labels.count_total`).
- **Color**: “dominant color” using `derived_labels.dominant_color`.
- **Shape**: “majority shape” using `derived_labels.majority_shape`.

### 6.3 Optional *neutral* routing prompts (for unspecified-task tests)

For any split, include:

```json
{
  "image_id": "img_B_000045",
  "split": "B",
  "task": "neutral",
  "prompt_template_id": "neutral_v1",
  "prompt_text": "Answer briefly.",
  "answer_space": null,
  "ground_truth": null
}
```

These are used to test whether adding a **task vector** routes the model to digits/colors/shapes without explicit instruction.

---

## 7) Balancing & quotas (agent must enforce)

- **Counts (per split)**: each count 1–6 must appear within ±2% of uniform.
- **Colors (Split A images)**: each color in **25% ±2%** of images.
- **Shapes (Split A images)**: each shape in **33% ±2%** of images.
- **Co-occurrence (Split B)**: for each color–shape pair, frequency within **(1 / 12) ±5%** of total images (since 4 colors × 3 shapes = 12 combos).
- **Seeds**: `seed = base_seed + image_index`; write `seed` in `images.jsonl` to ensure exact reproducibility.

---

## 8) Validation rules (write a small checker)

Per image:
- Split A must have: `len(unique(colors)) == 1` and `len(unique(shapes)) == 1`.
- Enforce non-overlap if `scene_constraints.non_overlap = true` (check all pairs using effective radii).
- Edge margin ≥ `min_edge_margin_px` for every object.
- Recompute `dominant_color` and `majority_shape` from polygons and verify against `derived_labels`.
- Ensure `bbox_xywh` encloses the polygon with a tolerance of **≤ 1 px**.
- Ensure PNG exists at `render.file` and has **256×256** resolution.

Global:
- Check balancing quotas (Section 7). Populate `manifests/stats.json` with counts and pass/fail.

---

## 9) Recommended sizes

- Split A: **500** images (→ 12,000 QA lines in `questions.jsonl` for matched triples).
- Split B: **500** images (+ optional ~2,000–6,000 QA lines if you add evaluations).
- Split C: **500** images (+ optional QA).

If compute-limited, you can scale down to A=2k, B=1k, C=500 and still get usable vectors.

---

## 10) Notes for the agent implementing the renderer

- Use **matplotlib** artists: `Circle`, `Rectangle`, and `Polygon` (equilateral triangle points rotated by `rotation_deg`). Fill with the specified hex colors, **no stroke** (or stroke width ≤ 1 px, same color).
- Convert floating centers/sizes to artists; do **not** quantize early—render at float precision, save as PNG.
- When computing `area_px`, rasterize the polygon mask or use analytic formulas (be consistent across shapes).
- To place objects without overlap: rejection sample centers within `[min+margin, max-margin]`, and check pairwise distances using **effective radii** (Section 1).
- Log all random draws in a debug log when `DEBUG=1` (seed, chosen shapes/colors/sizes, placement attempts).

---

## 11) Versioning

- Add `generator_version` (string) to every `images.jsonl` row.
- Start with `"shapes-spec-v1.0"` and bump when any schema or rendering rule changes.

---

## 12) Example minimal lines

**images.jsonl**
```json
{"image_id":"img_A_000001","split":"A","seed":73001,"canvas":{"width":256,"height":256,"bg_color":"#FFFFFF"},"stress":{},"objects":[{"id":"o1","shape":"square","color":"blue","center":{"x":72.3,"y":180.5},"size":{"radius":null,"side":36.0},"rotation_deg":0.0,"z":0,"bbox_xywh":[54.3,162.5,36.0,36.0],"area_px":1296.0},{"id":"o2","shape":"square","color":"blue","center":{"x":180.2,"y":84.7},"size":{"radius":null,"side":28.0},"rotation_deg":0.0,"z":1,"bbox_xywh":[166.2,70.7,28.0,28.0],"area_px":784.0}],"scene_constraints":{"min_edge_margin_px":6.0,"non_overlap":true,"sep_buffer_px":2.0,"effective_radius_formula":"see spec v1.0"},"derived_labels":{"count_total":2,"counts_by_color":{"red":0,"green":0,"blue":2,"yellow":0},"counts_by_shape":{"circle":0,"square":2,"triangle":0},"counts_by_color_shape":{"blue":{"circle":0,"square":2,"triangle":0}},"dominant_color_rule":"pixel_area","dominant_color":"blue","majority_shape_rule":"count_then_area_then_leftmost","majority_shape":"square"},"render":{"matplotlib":{"backend":"Agg","dpi":100,"antialias":true},"file":"images/A_factor_isolated/img_A_000001.png"},"generator_version":"shapes-spec-v1.0"}
```

**questions.jsonl**
```json
{"image_id":"img_A_000001","split":"A","task":"count","prompt_template_id":"count_v1","prompt_text":"Answer with one digit (0–9) only. How many objects are there?","answer_space":["0","1","2","3","4","5","6","7","8","9"],"ground_truth":"2","answer_rule":"exact_digit","answer_bottleneck_token_index":0}
{"image_id":"img_A_000001","split":"A","task":"color","prompt_template_id":"color_v1","prompt_text":"Answer with one word only from {red, green, blue, yellow}. What color are the objects?","answer_space":["red","green","blue","yellow"],"ground_truth":"blue","answer_rule":"exact_match","answer_bottleneck_token_index":0}
{"image_id":"img_A_000001","split":"A","task":"shape","prompt_template_id":"shape_v1","prompt_text":"Answer with one word only from {circle, square, triangle}. What shape are the objects?","answer_space":["circle","square","triangle"],"ground_truth":"square","answer_rule":"exact_match","answer_bottleneck_token_index":0}
```

---

## 13) Deliverables

- `images/` PNGs for all splits.
- `manifests/images.jsonl` and `manifests/questions.jsonl` per above schema.
- `manifests/stats.json` with split sizes, class frequencies, and validation outcomes.
- `README.md` with generator command, dependencies, and how to reproduce with a base seed.

---

**End of spec (v1.0).**