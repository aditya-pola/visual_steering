#!/usr/bin/env python3
"""
MLLM Task-Vector Dataset Generator (Matplotlib 2D Shapes)

Generates a dataset of 2D shape images following the exact specification
for learning single-dimension task vectors for count, color, and shape.
"""

import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Rectangle, Polygon
from collections import defaultdict, Counter

# Configuration constants from spec
CANVAS_SIZE = 256
COLORS = {
    'red': '#FF3B30',
    'green': '#34C759',
    'blue': '#007AFF',
    'yellow': '#FFCC00'
}

SHAPES = ['circle', 'square', 'triangle']
COLOR_NAMES = ['red', 'green', 'blue', 'yellow']

# Size ranges (inclusive)
SIZE_RANGES = {
    'circle': (14, 28),  # radius
    'square': (24, 44),  # side
    'triangle': (26, 48)  # side (equilateral)
}

MIN_EDGE_MARGIN = 6
SEP_BUFFER = 2

class DatasetGenerator:
    def __init__(self, base_seed: int = 42, output_dir: str = "data"):
        self.base_seed = base_seed
        self.output_dir = Path(output_dir)
        self.image_counter = 0
        
        # Create directory structure
        self.setup_directories()
        
        # Validation tracking
        self.validation_errors = []
        
    def setup_directories(self):
        """Create the required directory structure."""
        dirs = [
            self.output_dir / "images" / "A_factor_isolated",
            self.output_dir / "images" / "B_mixed", 
            self.output_dir / "images" / "C_stress",
            self.output_dir / "manifests"
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
            
    def get_effective_radius(self, shape: str, size: float) -> float:
        """Calculate effective radius for overlap detection."""
        if shape == 'circle':
            return size  # size is radius
        elif shape == 'square':
            return size / math.sqrt(2)  # circumscribed circle
        elif shape == 'triangle':
            return size / math.sqrt(3)  # circumscribed circle for equilateral
        else:
            raise ValueError(f"Unknown shape: {shape}")
            
    def check_overlap(self, center1: Tuple[float, float], radius1: float, 
                     center2: Tuple[float, float], radius2: float) -> bool:
        """Check if two objects overlap using effective radii."""
        dist = math.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        return dist < radius1 + radius2 + SEP_BUFFER
        
    def is_position_valid(self, center: Tuple[float, float], radius: float, 
                         existing_objects: List[Dict]) -> bool:
        """Check if position is valid (margins + no overlap)."""
        x, y = center
        
        # Check margins
        if (x - radius < MIN_EDGE_MARGIN or x + radius > CANVAS_SIZE - MIN_EDGE_MARGIN or
            y - radius < MIN_EDGE_MARGIN or y + radius > CANVAS_SIZE - MIN_EDGE_MARGIN):
            return False
            
        # Check overlaps with existing objects
        for obj in existing_objects:
            obj_center = (obj['center']['x'], obj['center']['y'])
            obj_radius = self.get_effective_radius(obj['shape'], 
                                                  obj['size']['radius'] if obj['shape'] == 'circle' else obj['size']['side'])
            if self.check_overlap(center, radius, obj_center, obj_radius):
                return False
                
        return True
        
    def place_object(self, shape: str, color: str, existing_objects: List[Dict], 
                    rng: random.Random, max_attempts: int = 1000) -> Optional[Dict]:
        """Place a single object without overlap."""
        size_min, size_max = SIZE_RANGES[shape]
        
        for attempt in range(max_attempts):
            # Generate random size and position
            if shape == 'circle':
                radius = rng.uniform(size_min, size_max)
                side = None
            else:
                radius = None
                side = rng.uniform(size_min, size_max)
                
            size_value = radius if radius is not None else side
            effective_radius = self.get_effective_radius(shape, size_value)
            
            # Random position
            min_coord = MIN_EDGE_MARGIN + effective_radius
            max_coord = CANVAS_SIZE - MIN_EDGE_MARGIN - effective_radius
            
            if min_coord >= max_coord:
                continue  # Object too large
                
            center_x = rng.uniform(min_coord, max_coord)
            center_y = rng.uniform(min_coord, max_coord)
            center = (center_x, center_y)
            
            if self.is_position_valid(center, effective_radius, existing_objects):
                # Valid position found
                rotation_deg = rng.uniform(0, 360) if shape == 'triangle' else 0.0
                
                # Calculate bounding box and area
                bbox, area = self.calculate_bbox_area(shape, center, size_value, rotation_deg)
                
                obj = {
                    "id": f"o{len(existing_objects) + 1}",
                    "shape": shape,
                    "color": color,
                    "center": {"x": center_x, "y": center_y},
                    "size": {
                        "radius": radius,
                        "side": side
                    },
                    "rotation_deg": rotation_deg,
                    "z": len(existing_objects),  # Will be randomized later
                    "bbox_xywh": bbox,
                    "area_px": area
                }
                return obj
                
        return None  # Failed to place
        
    def calculate_bbox_area(self, shape: str, center: Tuple[float, float], 
                          size: float, rotation_deg: float) -> Tuple[List[float], float]:
        """Calculate bounding box and area for a shape."""
        x, y = center
        
        if shape == 'circle':
            radius = size
            bbox = [x - radius, y - radius, 2 * radius, 2 * radius]
            area = math.pi * radius * radius
            
        elif shape == 'square':
            side = size
            half_side = side / 2
            bbox = [x - half_side, y - half_side, side, side]
            area = side * side
            
        elif shape == 'triangle':
            side = size
            # Equilateral triangle inscribed in circle
            radius = side / math.sqrt(3)
            # For bounding box, we need to consider rotation
            # Approximate with circumscribed circle for simplicity
            bbox = [x - radius, y - radius, 2 * radius, 2 * radius]
            area = (math.sqrt(3) / 4) * side * side
            
        return bbox, area
        
    def generate_triangle_points(self, center: Tuple[float, float], side: float, 
                               rotation_deg: float) -> List[Tuple[float, float]]:
        """Generate vertices of an equilateral triangle."""
        x, y = center
        radius = side / math.sqrt(3)  # Circumradius
        
        points = []
        for i in range(3):
            angle = math.radians(rotation_deg + i * 120)
            px = x + radius * math.cos(angle)
            py = y + radius * math.sin(angle)
            points.append((px, py))
            
        return points
        
    def render_image(self, objects: List[Dict], image_path: str):
        """Render the image using matplotlib."""
        fig, ax = plt.subplots(figsize=(2.56, 2.56), dpi=100)
        ax.set_xlim(0, CANVAS_SIZE)
        ax.set_ylim(0, CANVAS_SIZE)
        ax.set_aspect('equal')
        ax.axis('off')
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        
        # Sort objects by z-order
        sorted_objects = sorted(objects, key=lambda obj: obj['z'])
        
        for obj in sorted_objects:
            color = COLORS[obj['color']]
            center = (obj['center']['x'], obj['center']['y'])
            
            if obj['shape'] == 'circle':
                radius = obj['size']['radius']
                circle = Circle(center, radius, facecolor=color, edgecolor=color, linewidth=0)
                ax.add_patch(circle)
                
            elif obj['shape'] == 'square':
                side = obj['size']['side']
                x, y = center
                # Rectangle with bottom-left corner
                rect = Rectangle((x - side/2, y - side/2), side, side, 
                               facecolor=color, edgecolor=color, linewidth=0)
                ax.add_patch(rect)
                
            elif obj['shape'] == 'triangle':
                side = obj['size']['side']
                rotation = obj['rotation_deg']
                points = self.generate_triangle_points(center, side, rotation)
                triangle = Polygon(points, facecolor=color, edgecolor=color, linewidth=0)
                ax.add_patch(triangle)
                
        plt.savefig(image_path, bbox_inches='tight', pad_inches=0, 
                   facecolor='white', dpi=100)
        plt.close()
        
    def calculate_derived_labels(self, objects: List[Dict]) -> Dict:
        """Calculate derived labels for an image."""
        count_total = len(objects)
        
        counts_by_color = Counter(obj['color'] for obj in objects)
        counts_by_shape = Counter(obj['shape'] for obj in objects)
        
        # Ensure all colors and shapes are represented
        for color in COLOR_NAMES:
            if color not in counts_by_color:
                counts_by_color[color] = 0
                
        for shape in SHAPES:
            if shape not in counts_by_shape:
                counts_by_shape[shape] = 0
                
        # Color-shape combinations
        counts_by_color_shape = defaultdict(lambda: defaultdict(int))
        total_area_by_color = defaultdict(float)
        
        for obj in objects:
            color = obj['color']
            shape = obj['shape'] 
            area = obj['area_px']
            counts_by_color_shape[color][shape] += 1
            total_area_by_color[color] += area
            
        # Convert to regular dicts for JSON serialization
        counts_by_color_shape = {
            color: dict(shape_counts) 
            for color, shape_counts in counts_by_color_shape.items()
        }
        
        # Dominant color (by pixel area)
        if total_area_by_color:
            dominant_color = max(total_area_by_color.items(), key=lambda x: x[1])[0]
        else:
            dominant_color = None
            
        # Majority shape (by count, then area, then leftmost)
        if objects:
            shape_counts = counts_by_shape
            max_count = max(shape_counts.values())
            tied_shapes = [shape for shape, count in shape_counts.items() if count == max_count]
            
            if len(tied_shapes) == 1:
                majority_shape = tied_shapes[0]
            else:
                # Tie-break by area
                area_by_shape = defaultdict(float)
                for obj in objects:
                    if obj['shape'] in tied_shapes:
                        area_by_shape[obj['shape']] += obj['area_px']
                        
                max_area = max(area_by_shape.values())
                area_tied = [shape for shape, area in area_by_shape.items() if area == max_area]
                
                if len(area_tied) == 1:
                    majority_shape = area_tied[0]
                else:
                    # Tie-break by leftmost
                    leftmost_x = float('inf')
                    leftmost_shape = None
                    for obj in objects:
                        if obj['shape'] in area_tied and obj['center']['x'] < leftmost_x:
                            leftmost_x = obj['center']['x']
                            leftmost_shape = obj['shape']
                    majority_shape = leftmost_shape
        else:
            majority_shape = None
            
        return {
            "count_total": count_total,
            "counts_by_color": dict(counts_by_color),
            "counts_by_shape": dict(counts_by_shape),
            "counts_by_color_shape": counts_by_color_shape,
            "dominant_color_rule": "pixel_area",
            "dominant_color": dominant_color,
            "majority_shape_rule": "count_then_area_then_leftmost",
            "majority_shape": majority_shape
        }
        
    def generate_split_a_image(self, rng: random.Random) -> Dict:
        """Generate a factor-isolated image (same color and shape)."""
        # Choose single color and shape
        color = rng.choice(COLOR_NAMES)
        shape = rng.choice(SHAPES)
        count = rng.randint(1, 6)
        
        objects = []
        for _ in range(count):
            obj = self.place_object(shape, color, objects, rng)
            if obj is None:
                # Failed to place, try again with fewer objects
                break
            objects.append(obj)
            
        if not objects:
            raise Exception("Failed to place any objects")
            
        # Randomize z-order
        z_order = list(range(len(objects)))
        rng.shuffle(z_order)
        for i, obj in enumerate(objects):
            obj['z'] = z_order[i]
            
        return objects
        
    def generate_split_b_image(self, rng: random.Random) -> Dict:
        """Generate a mixed-attribute image."""
        count = rng.randint(1, 6)
        
        objects = []
        for _ in range(count):
            # Random color and shape for each object
            color = rng.choice(COLOR_NAMES)
            shape = rng.choice(SHAPES)
            
            obj = self.place_object(shape, color, objects, rng)
            if obj is None:
                break
            objects.append(obj)
            
        if not objects:
            raise Exception("Failed to place any objects")
            
        # Randomize z-order
        z_order = list(range(len(objects)))
        rng.shuffle(z_order)
        for i, obj in enumerate(objects):
            obj['z'] = z_order[i]
            
        return objects
        
    def generate_split_c_image(self, rng: random.Random) -> Tuple[Dict, Dict]:
        """Generate a stress/OOD image with specified stress conditions."""
        # For now, implement basic version - can be extended
        stress_config = {
            "occlusion_target": 0.0,
            "low_contrast": False,
            "color_confusion": False,
            "size_extreme": False
        }
        
        # Randomly pick stress conditions
        stress_type = rng.choice(['occlusion_30', 'low_contrast', 'color_confusion', 'size_extreme'])
        
        if stress_type == 'occlusion_30':
            stress_config["occlusion_target"] = 0.3
        elif stress_type == 'low_contrast':
            stress_config["low_contrast"] = True
        elif stress_type == 'color_confusion':
            stress_config["color_confusion"] = True
        elif stress_type == 'size_extreme':
            stress_config["size_extreme"] = True
            
        # Generate objects similar to split B but allow overlaps for occlusion
        count = rng.randint(1, 6)
        objects = []
        
        for _ in range(count):
            color = rng.choice(COLOR_NAMES)
            shape = rng.choice(SHAPES)
            
            # For occlusion, relax overlap constraints
            if stress_config["occlusion_target"] > 0:
                obj = self.place_object_with_overlap(shape, color, objects, rng, stress_config)
            else:
                obj = self.place_object(shape, color, objects, rng)
                
            if obj is None:
                break
            objects.append(obj)
            
        if not objects:
            raise Exception("Failed to place any objects")
            
        # Randomize z-order
        z_order = list(range(len(objects)))
        rng.shuffle(z_order)
        for i, obj in enumerate(objects):
            obj['z'] = z_order[i]
            
        return objects, stress_config
        
    def place_object_with_overlap(self, shape: str, color: str, existing_objects: List[Dict], 
                                rng: random.Random, stress_config: Dict, max_attempts: int = 500) -> Optional[Dict]:
        """Place object allowing controlled overlaps for stress conditions."""
        # This is a simplified implementation - can be made more sophisticated
        size_min, size_max = SIZE_RANGES[shape]
        
        if stress_config.get("size_extreme", False):
            # Push to extremes
            if rng.random() < 0.5:
                size_value = size_min  # Very small
            else:
                size_value = size_max  # Very large
        else:
            size_value = rng.uniform(size_min, size_max)
            
        for attempt in range(max_attempts):
            if shape == 'circle':
                radius = size_value
                side = None
            else:
                radius = None
                side = size_value
                
            effective_radius = self.get_effective_radius(shape, size_value)
            
            # Allow wider placement range
            min_coord = MIN_EDGE_MARGIN + effective_radius * 0.5
            max_coord = CANVAS_SIZE - MIN_EDGE_MARGIN - effective_radius * 0.5
            
            if min_coord >= max_coord:
                continue
                
            center_x = rng.uniform(min_coord, max_coord)
            center_y = rng.uniform(min_coord, max_coord)
            center = (center_x, center_y)
            
            # For occlusion, allow some overlap
            if stress_config.get("occlusion_target", 0) > 0:
                # Just check basic bounds, allow overlaps
                if (center_x - effective_radius >= 0 and center_x + effective_radius <= CANVAS_SIZE and
                    center_y - effective_radius >= 0 and center_y + effective_radius <= CANVAS_SIZE):
                    rotation_deg = rng.uniform(0, 360) if shape == 'triangle' else 0.0
                    bbox, area = self.calculate_bbox_area(shape, center, size_value, rotation_deg)
                    
                    obj = {
                        "id": f"o{len(existing_objects) + 1}",
                        "shape": shape,
                        "color": color,
                        "center": {"x": center_x, "y": center_y},
                        "size": {"radius": radius, "side": side},
                        "rotation_deg": rotation_deg,
                        "z": len(existing_objects),
                        "bbox_xywh": bbox,
                        "area_px": area
                    }
                    return obj
            else:
                # Regular placement
                if self.is_position_valid(center, effective_radius, existing_objects):
                    rotation_deg = rng.uniform(0, 360) if shape == 'triangle' else 0.0
                    bbox, area = self.calculate_bbox_area(shape, center, size_value, rotation_deg)
                    
                    obj = {
                        "id": f"o{len(existing_objects) + 1}",
                        "shape": shape,
                        "color": color,
                        "center": {"x": center_x, "y": center_y},
                        "size": {"radius": radius, "side": side},
                        "rotation_deg": rotation_deg,
                        "z": len(existing_objects),
                        "bbox_xywh": bbox,
                        "area_px": area
                    }
                    return obj
                    
        return None
        
    def create_image_entry(self, image_id: str, split: str, seed: int, objects: List[Dict], 
                          stress_config: Optional[Dict] = None) -> Dict:
        """Create an image entry for the manifest."""
        derived_labels = self.calculate_derived_labels(objects)
        
        return {
            "image_id": image_id,
            "split": split,
            "seed": seed,
            "canvas": {"width": CANVAS_SIZE, "height": CANVAS_SIZE, "bg_color": "#FFFFFF"},
            "stress": stress_config if stress_config else {},
            "objects": objects,
            "scene_constraints": {
                "min_edge_margin_px": float(MIN_EDGE_MARGIN),
                "non_overlap": split in ['A', 'B'],  # Split C may allow overlap
                "sep_buffer_px": float(SEP_BUFFER),
                "effective_radius_formula": "see spec v1.0"
            },
            "derived_labels": derived_labels,
            "render": {
                "matplotlib": {
                    "backend": "Agg",
                    "dpi": 100,
                    "antialias": True
                },
                "file": f"images/{split}_{'factor_isolated' if split == 'A' else 'mixed' if split == 'B' else 'stress'}/{image_id}.png"
            },
            "generator_version": "shapes-spec-v1.0"
        }
        
    def create_question_entries(self, image_id: str, split: str, derived_labels: Dict) -> List[Dict]:
        """Create question entries for an image."""
        questions = []
        
        # Count question
        questions.append({
            "image_id": image_id,
            "split": split,
            "task": "count",
            "prompt_template_id": "count_v1",
            "prompt_text": "Answer with one digit (0–9) only. How many objects are there?",
            "answer_space": ["0","1","2","3","4","5","6","7","8","9"],
            "ground_truth": str(derived_labels["count_total"]),
            "answer_rule": "exact_digit",
            "answer_bottleneck_token_index": 0
        })
        
        # Color question  
        questions.append({
            "image_id": image_id,
            "split": split,
            "task": "color",
            "prompt_template_id": "color_v1",
            "prompt_text": "Answer with one word only from {red, green, blue, yellow}. What color are the objects?",
            "answer_space": ["red", "green", "blue", "yellow"],
            "ground_truth": derived_labels["dominant_color"],
            "answer_rule": "exact_match",
            "answer_bottleneck_token_index": 0
        })
        
        # Shape question
        questions.append({
            "image_id": image_id,
            "split": split,
            "task": "shape", 
            "prompt_template_id": "shape_v1",
            "prompt_text": "Answer with one word only from {circle, square, triangle}. What shape are the objects?",
            "answer_space": ["circle", "square", "triangle"],
            "ground_truth": derived_labels["majority_shape"],
            "answer_rule": "exact_match",
            "answer_bottleneck_token_index": 0
        })
        
        return questions
        
    def generate_dataset(self, split_sizes: Dict[str, int] = None):
        """Generate the complete dataset."""
        if split_sizes is None:
            split_sizes = {'A': 500, 'B': 500, 'C': 500}
            
        print("Generating MLLM Task-Vector Dataset...")
        print(f"Split sizes: {split_sizes}")
        
        all_image_entries = []
        all_question_entries = []
        
        # Generate Split A (Factor-isolated)
        print("\nGenerating Split A (Factor-isolated)...")
        for i in range(split_sizes['A']):
            seed = self.base_seed + i
            rng = random.Random(seed)
            
            image_id = f"img_A_{i+1:06d}"
            
            try:
                objects = self.generate_split_a_image(rng)
                
                # Create image entry
                image_entry = self.create_image_entry(image_id, 'A', seed, objects)
                all_image_entries.append(image_entry)
                
                # Render image
                image_path = self.output_dir / image_entry['render']['file']
                self.render_image(objects, str(image_path))
                
                # Create questions (only for Split A as per spec)
                questions = self.create_question_entries(image_id, 'A', image_entry['derived_labels'])
                all_question_entries.extend(questions)
                
                if (i + 1) % 100 == 0:
                    print(f"  Generated {i+1}/{split_sizes['A']} images")
                    
            except Exception as e:
                print(f"  Error generating {image_id}: {e}")
                self.validation_errors.append(f"Split A {image_id}: {e}")
                
        # Generate Split B (Mixed-attribute)
        print("\nGenerating Split B (Mixed-attribute)...")
        for i in range(split_sizes['B']):
            seed = self.base_seed + split_sizes['A'] + i
            rng = random.Random(seed)
            
            image_id = f"img_B_{i+1:06d}"
            
            try:
                objects = self.generate_split_b_image(rng)
                
                # Create image entry
                image_entry = self.create_image_entry(image_id, 'B', seed, objects)
                all_image_entries.append(image_entry)
                
                # Render image
                image_path = self.output_dir / image_entry['render']['file']
                self.render_image(objects, str(image_path))
                
                if (i + 1) % 100 == 0:
                    print(f"  Generated {i+1}/{split_sizes['B']} images")
                    
            except Exception as e:
                print(f"  Error generating {image_id}: {e}")
                self.validation_errors.append(f"Split B {image_id}: {e}")
                
        # Generate Split C (Stress/OOD)
        print("\nGenerating Split C (Stress/OOD)...")
        for i in range(split_sizes['C']):
            seed = self.base_seed + split_sizes['A'] + split_sizes['B'] + i
            rng = random.Random(seed)
            
            image_id = f"img_C_{i+1:06d}"
            
            try:
                objects, stress_config = self.generate_split_c_image(rng)
                
                # Create image entry
                image_entry = self.create_image_entry(image_id, 'C', seed, objects, stress_config)
                all_image_entries.append(image_entry)
                
                # Render image
                image_path = self.output_dir / image_entry['render']['file']
                self.render_image(objects, str(image_path))
                
                if (i + 1) % 100 == 0:
                    print(f"  Generated {i+1}/{split_sizes['C']} images")
                    
            except Exception as e:
                print(f"  Error generating {image_id}: {e}")
                self.validation_errors.append(f"Split C {image_id}: {e}")
                
        # Save manifests
        print("\nSaving manifests...")
        
        # Save images.jsonl
        images_file = self.output_dir / "manifests" / "images.jsonl"
        with open(images_file, 'w') as f:
            for entry in all_image_entries:
                f.write(json.dumps(entry) + '\n')
                
        # Save questions.jsonl
        questions_file = self.output_dir / "manifests" / "questions.jsonl"
        with open(questions_file, 'w') as f:
            for entry in all_question_entries:
                f.write(json.dumps(entry) + '\n')
                
        # Generate and save stats
        stats = self.generate_stats(all_image_entries, all_question_entries)
        stats_file = self.output_dir / "manifests" / "stats.json"
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
            
        # Create README
        self.create_readme(split_sizes)
        
        print(f"\nDataset generation complete!")
        print(f"Generated {len(all_image_entries)} images")
        print(f"Generated {len(all_question_entries)} questions")
        print(f"Output directory: {self.output_dir}")
        
        if self.validation_errors:
            print(f"\nWarning: {len(self.validation_errors)} validation errors occurred:")
            for error in self.validation_errors[:5]:  # Show first 5
                print(f"  {error}")
            if len(self.validation_errors) > 5:
                print(f"  ... and {len(self.validation_errors) - 5} more")
                
    def generate_stats(self, image_entries: List[Dict], question_entries: List[Dict]) -> Dict:
        """Generate statistics for validation."""
        stats = {
            "total_images": len(image_entries),
            "total_questions": len(question_entries),
            "split_sizes": {},
            "validation_errors": self.validation_errors,
            "balancing_check": {}
        }
        
        # Split sizes
        split_counts = Counter(entry['split'] for entry in image_entries)
        stats["split_sizes"] = dict(split_counts)
        
        # Balancing analysis for Split A
        split_a_images = [entry for entry in image_entries if entry['split'] == 'A']
        
        if split_a_images:
            # Count distribution
            count_dist = Counter(entry['derived_labels']['count_total'] for entry in split_a_images)
            
            # Color distribution (for Split A)
            color_dist = Counter()
            for entry in split_a_images:
                for obj in entry['objects']:
                    color_dist[obj['color']] += 1
                    
            # Shape distribution (for Split A)
            shape_dist = Counter()
            for entry in split_a_images:
                for obj in entry['objects']:
                    shape_dist[obj['shape']] += 1
                    
            stats["balancing_check"]["split_a"] = {
                "count_distribution": dict(count_dist),
                "color_distribution": dict(color_dist),
                "shape_distribution": dict(shape_dist)
            }
            
        return stats
        
    def create_readme(self, split_sizes: Dict[str, int]):
        """Create README with generation info."""
        readme_content = f"""# MLLM Task-Vector Dataset

This dataset was generated according to the shapes-spec-v1.0 specification.

## Generation Details

- **Generator**: generate_dataset.py
- **Base seed**: {self.base_seed}
- **Split sizes**: {split_sizes}
- **Canvas size**: {CANVAS_SIZE}x{CANVAS_SIZE} pixels
- **Colors**: {list(COLORS.keys())}
- **Shapes**: {SHAPES}

## Reproducing the Dataset

To regenerate this exact dataset:

```bash
python generate_dataset.py --base_seed {self.base_seed} --output_dir {self.output_dir}
```

## Dataset Structure

```
{self.output_dir}/
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
"""

        readme_path = self.output_dir / "README.md"
        with open(readme_path, 'w') as f:
            f.write(readme_content)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate MLLM Task-Vector Dataset")
    parser.add_argument("--base_seed", type=int, default=42, help="Base random seed")
    parser.add_argument("--output_dir", type=str, default="data", help="Output directory")
    parser.add_argument("--split_a_size", type=int, default=500, help="Split A size")
    parser.add_argument("--split_b_size", type=int, default=500, help="Split B size") 
    parser.add_argument("--split_c_size", type=int, default=500, help="Split C size")
    
    args = parser.parse_args()
    
    split_sizes = {
        'A': args.split_a_size,
        'B': args.split_b_size, 
        'C': args.split_c_size
    }
    
    generator = DatasetGenerator(base_seed=args.base_seed, output_dir=args.output_dir)
    generator.generate_dataset(split_sizes)


if __name__ == "__main__":
    main()
