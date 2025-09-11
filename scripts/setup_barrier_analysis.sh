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


# Make scripts executable
echo "🔑 Making analysis scripts executable..."
chmod +x analysis/run_analysis.py
chmod +x analysis/simple_barrier_test.py
chmod +x analysis/barrier_representation_analysis.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "🚀 To run the analysis:"
echo "   Run analysis:   python analysis/run_analysis.py"
echo "   Custom config:  python analysis/barrier_representation_analysis.py --help"
echo "   With options:   python analysis/run_analysis.py --num_episodes 5 --model Qwen/Qwen2.5-7B-Instruct"
echo ""
echo "📊 Results will be saved to:"
echo "   - results/barrier_analysis/"
echo ""
echo "📖 For detailed documentation:"
echo "   - See analysis/README.md"