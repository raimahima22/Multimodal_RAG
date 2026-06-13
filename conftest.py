"""
conftest.py — shared pytest configuration.
Ensures the project root is on sys.path so 'src.*' imports resolve
without installing the package.
"""
import sys
from pathlib import Path

# Add project root (one level above /tests) to the import path
sys.path.insert(0, str(Path(__file__).parent.parent))