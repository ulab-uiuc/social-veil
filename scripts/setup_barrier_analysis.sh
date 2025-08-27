#!/bin/bash

# Setup script for barrier representation analysis

echo "🔧 Setting up Barrier Representation Analysis"
echo "============================================="

# Check if we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo "❌ Please run this script from the project root directory"
    exit 1
fi

# Install analysis requirements
echo "📦 Installing analysis dependencies..."
pip install -r analysis/requirements.txt

# Create results directory
echo "📁 Creating results directory..."
mkdir -p results/barrier_analysis
mkdir -p results/simple_barrier_test

# Check if sample episodes exist
if [ ! -f "data/episode_sample.jsonl" ]; then
    echo "⚠️  Warning: data/episode_sample.jsonl not found"
    echo "   You may need to create sample episodes first"
    echo "   or use a different episodes file"
fi

# Make scripts executable
echo "🔑 Making analysis scripts executable..."
chmod +x analysis/run_analysis.py
chmod +x analysis/simple_barrier_test.py
chmod +x analysis/barrier_representation_analysis.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To run the analysis:"
echo "   Quick test:     python analysis/run_analysis.py --mode simple"
echo "   Full analysis:  python analysis/run_analysis.py --mode full"
echo "   Custom config:  python analysis/barrier_representation_analysis.py --help"
echo ""
echo "📊 Results will be saved to:"
echo "   - results/simple_barrier_test/"
echo "   - results/barrier_analysis/"
echo ""
echo "📖 For detailed documentation:"
echo "   - See analysis/README.md"