"""Shared pytest fixtures."""
import sys
from pathlib import Path

# Make sure src/ is importable when running tests from project root
sys.path.insert(0, str(Path(__file__).parent.parent))
