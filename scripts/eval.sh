#!/bin/bash

# MLLM Task-Vector Dataset Evaluation Runner
# This script provides easy commands for running evaluations

set -e

echo "🎯 MLLM Task-Vector Dataset Evaluation Suite"
echo "=============================================="

# Check if conda environment is active
if [[ "$CONDA_DEFAULT_ENV" != "visteer" ]]; then
    echo "❌ Please activate the visteer environment first:"
    echo "   conda activate visteer"
    exit 1
fi

# Function to run evaluation
run_eval() {
    local desc=$1
    local cmd=$2
    
    echo ""
    echo "🚀 $desc"
    echo "Command: $cmd"
    echo "----------------------------------------"
    
    eval $cmd
    
    if [ $? -eq 0 ]; then
        echo "✅ $desc completed successfully!"
    else
        echo "❌ $desc failed!"
        exit 1
    fi
}

# Parse command line arguments
case "${1:-help}" in
    "quick")
        run_eval "Quick Test (10 samples, Split A only)" \
                "python evaluate_baseline.py --quick"
        ;;
        
    "split-a")
        run_eval "Split A Evaluation (Factor-isolated, 500 samples)" \
                "python evaluate_baseline.py --splits A"
        ;;
        
    "full")
        run_eval "Full Dataset Evaluation (all splits, 1500 samples)" \
                "python evaluate_baseline.py"
        ;;
        
    "custom")
        if [ -z "$2" ]; then
            echo "❌ Please provide a model name for custom evaluation"
            echo "Example: ./eval.sh custom 'google/paligemma-3b-mix-448'"
            exit 1
        fi
        run_eval "Custom Model Evaluation: $2" \
                "python evaluate_baseline.py --model '$2' --quick"
        ;;
        
    "generate")
        echo "🎨 Generating dataset..."
        case "${2:-full}" in
            "small")
                run_eval "Small Dataset Generation" \
                        "python generate_dataset.py --split_a_size 100 --split_b_size 100 --split_c_size 100"
                ;;
            "full")
                run_eval "Full Dataset Generation" \
                        "python generate_dataset.py"
                ;;
        esac
        ;;
        
    "help"|*)
        echo ""
        echo "📖 Usage: ./eval.sh <command>"
        echo ""
        echo "Commands:"
        echo "  quick      - Quick test (10 samples, Split A only)"
        echo "  split-a    - Evaluate Split A only (500 samples)"  
        echo "  full       - Full evaluation (all 1500 samples)"
        echo "  custom <model> - Evaluate custom model (quick test)"
        echo "  generate [small|full] - Generate dataset"
        echo "  help       - Show this help"
        echo ""
        echo "Examples:"
        echo "  ./eval.sh quick"
        echo "  ./eval.sh split-a"
        echo "  ./eval.sh custom 'google/paligemma-3b-mix-448'"
        echo "  ./eval.sh generate small"
        echo ""
        ;;
esac

echo ""
echo "🎯 Task-Vector Dataset Evaluation Complete!"
