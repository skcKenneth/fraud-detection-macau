"""
Setup script to download and prepare all data files
設置腳本以下載和準備所有數據文件
"""
import sys
import argparse
from pathlib import Path

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.kaggle_downloader import setup_kaggle_data
from data.user_profile_generator import setup_user_profiles
from config import DATA_DIR, TRANSACTIONS_FILE, USER_PROFILES_FILE

def main():
    parser = argparse.ArgumentParser(
        description='Setup data for Cross-Border Fraud Detection System'
    )
    parser.add_argument(
        '--local',
        action='store_true',
        help='Use local Kaggle data file (skip download)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force regenerate all data files'
    )
    parser.add_argument(
        '--users',
        type=int,
        default=100,
        help='Number of user profiles to generate (default: 100)'
    )
    
    args = parser.parse_args()
    
    print("="*70)
    print("Cross-Border Fraud Detection System - Data Setup")
    print("="*70)
    
    # Create data directory
    DATA_DIR.mkdir(exist_ok=True)
    print(f"\nData directory: {DATA_DIR}")
    
    success_count = 0
    total_steps = 2
    
    # Step 1: Setup transaction data
    print(f"\n{'='*70}")
    print("Step 1/2: Setup transaction data")
    print("="*70)
    
    # Always try to setup data (will generate sample if Kaggle not available)
    result = setup_kaggle_data(force_download=args.force)
    if result:
        success_count += 1
    else:
        print("\nFailed to setup transaction data")
    
    # Step 2: Setup user profiles
    print(f"\n{'='*70}")
    print("Step 2/2: Setup user behavior profiles")
    print("="*70)
    
    result = setup_user_profiles(
        n_users=args.users,
        force_regenerate=args.force
    )
    if result:
        success_count += 1
    
    # Summary
    print(f"\n{'='*70}")
    print("Setup completed")
    print("="*70)
    print(f"Successfully completed: {success_count}/{total_steps} steps")
    
    if success_count == total_steps:
        print("\nAll data is ready!")
        print("\nNext step:")
        print("  Run application: streamlit run app.py")
    else:
        print("\nSome steps failed, please check the error messages above")
        return 1
    
    # Verify files
    print(f"\nData files:")
    if TRANSACTIONS_FILE.exists():
        print(f"  + {TRANSACTIONS_FILE}")
    else:
        print(f"  - {TRANSACTIONS_FILE}")
    
    if USER_PROFILES_FILE.exists():
        print(f"  + {USER_PROFILES_FILE}")
    else:
        print(f"  - {USER_PROFILES_FILE}")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
