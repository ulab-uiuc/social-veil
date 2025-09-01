#!/usr/bin/env python3
"""
Install datasets library for IQ tests
"""

import subprocess
import sys

def install_datasets():
    """Install datasets library if not available"""
    try:
        import datasets
        print("✅ datasets library already installed")
        return True
    except ImportError:
        print("📦 Installing datasets library...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "datasets", "huggingface_hub"])
            print("✅ datasets library installed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install datasets library: {e}")
            return False

if __name__ == "__main__":
    success = install_datasets()
    if not success:
        print("\n⚠️ Please install manually:")
        print("pip install datasets huggingface_hub")
        sys.exit(1)