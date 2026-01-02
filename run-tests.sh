#!/bin/bash
# Simple test runner script for Spark MLIR

set -e

# Navigate to the build directory
cd "$(dirname "$0")/build"

# Path to llvm-lit
LLVM_LIT=~/src/llvm-project/build/bin/llvm-lit

# Check if llvm-lit exists
if [ ! -f "$LLVM_LIT" ]; then
    echo "Error: llvm-lit not found at $LLVM_LIT"
    exit 1
fi

# Default to verbose mode
VERBOSE="-v"

# Parse command line arguments
case "${1:-all}" in
    all)
        echo "Running all tests..."
        $LLVM_LIT test/ $VERBOSE
        ;;
    spark)
        echo "Running Spark dialect tests..."
        $LLVM_LIT test/Spark/ $VERBOSE
        ;;
    bark)
        echo "Running bark operation test..."
        $LLVM_LIT test/Spark/bark.mlir $VERBOSE
        ;;
    -h|--help)
        echo "Usage: $0 [all|spark|bark]"
        echo ""
        echo "Options:"
        echo "  all   - Run all tests (default)"
        echo "  spark - Run only Spark dialect tests"
        echo "  bark  - Run only the bark operation test"
        echo "  -h    - Show this help message"
        exit 0
        ;;
    *)
        echo "Unknown option: $1"
        echo "Use -h for help"
        exit 1
        ;;
esac
