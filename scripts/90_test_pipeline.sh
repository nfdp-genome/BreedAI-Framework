#!/bin/bash
# File: 90_test_pipeline.sh
# Purpose: Test the array job setup

# Project paths - automatically detect BreedAI-Framework directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
SCRIPTS_DIR="$PROJECT_DIR/scripts"
TEST_DIR="$PROJECT_DIR/test_array"

echo "Testing array job setup..."

# Create test directory
mkdir -p $TEST_DIR
cd $SCRIPTS_DIR

# Make sure all scripts are executable
chmod +x *.py *.sh

# Test the benchmarking array preparation (dry run)
python 02a_phase1_train_validate_array.py --help

echo "Array job system ready!"
echo ""
echo "To run the complete pipeline:"
echo "1. cd $SCRIPTS_DIR"
echo "2. ./start_menu.sh          # Simple launcher (runs complete pipeline)"
echo "   OR"
echo "   ./01_pipeline_run_all.sh 4     # Run complete pipeline directly"