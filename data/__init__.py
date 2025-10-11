"""
Data module for fraud detection system
"""
from pathlib import Path

DATA_DIR = Path(__file__).parent

__all__ = ['kaggle_downloader', 'user_profile_generator', 'DATA_DIR']
