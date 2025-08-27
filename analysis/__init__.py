# Barrier Representation Analysis Module
"""
Analysis tools for studying communication barrier effects on model representations.

This module provides tools to:
1. Extract internal representations from language models
2. Analyze distribution shifts between baseline and barrier conditions
3. Visualize representational differences
4. Apply statistical tests to prove barrier effects

Main components:
- barrier_representation_analysis.py: Comprehensive analysis framework
- simple_barrier_test.py: Quick test for barrier effects
- utils.py: Helper functions and utilities
"""

__version__ = "1.0.0"
__author__ = "Social-Decipher Research Team"

from .barrier_representation_analysis import BarrierRepresentationAnalyzer

__all__ = ['BarrierRepresentationAnalyzer']