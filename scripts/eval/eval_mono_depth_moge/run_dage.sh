#!/bin/bash

# Script to run inference and evaluation on multiple benchmarks using the combined script
# Usage: bash ./experiments/eval_scripts/eval_mono_depth_moge/run_mogev2_pi3_combined_v2.sh <checkpoint_path> <output_dir>

CHECKPOINT=${1:-"checkpoints/model.pt"}
OUTPUT_DIR=${2:-"./eval_results/mono_depth_moge/dage"}
BENCHMARKS=("NYUv2" "ETH3D" "Sintel" "GSO" "KITTI" "Spring" "HAMMER" "iBims-1" "DDAD" "DIODE")


echo "Running inference and evaluation with:"
echo "Checkpoint: $CHECKPOINT"
echo "Output directory: $OUTPUT_DIR"
echo "Benchmarks: ${BENCHMARKS[@]}"
echo ""

# Create output directory
mkdir -p $OUTPUT_DIR

# Run processing on all benchmarks
echo "=== Processing Benchmarks ==="
for benchmark in "${BENCHMARKS[@]}"; do
    echo "Processing $benchmark ..."
    python evaluation/monodepth_moge/run_dage.py \
        --checkpoint "$CHECKPOINT" \
        --benchmark "$benchmark" \
        --output_dir "$OUTPUT_DIR" \
        --lr_max_size 252 \
        --seed 42
    
    if [ $? -eq 0 ]; then
        echo "✓ Processing completed for $benchmark"
    else
        echo "✗ Processing failed for $benchmark"
    fi
    echo ""
done

# Collect results
echo "=== Results Summary ==="
echo "Results saved in: $OUTPUT_DIR"
echo ""
echo "Individual benchmark results:"
for benchmark in "${BENCHMARKS[@]}"; do
    if [ -f "$OUTPUT_DIR/$benchmark/eval_metrics.json" ]; then
        echo "- $benchmark: $OUTPUT_DIR/$benchmark/eval_metrics.json"
        # Show summary metrics from JSON
        python -c "import json; data=json.load(open('$OUTPUT_DIR/$benchmark/eval_metrics.json')); print(f'  Mean metrics: {data[\"summary\"][\"mean_metrics\"]}'); print(f'  Samples processed: {data[\"summary\"][\"num_samples\"]}')" 2>/dev/null || echo "  (Could not parse metrics)"
    else
        echo "- $benchmark: FAILED"
    fi
done

echo ""
echo "All done!"
