"""
Data Loading and Preprocessing Utilities
Data Loading and Preprocessing Utilities
"""
import pandas as pd
import json
from pathlib import Path
from config import TRANSACTIONS_FILE, USER_PROFILES_FILE

def load_transactions(file_path=None, sample_size=None):
    """
    Load transaction data
    Load transaction data
    
    Args:
        file_path: Path to CSV file (default: from config)
        sample_size: Number of rows to sample (None = all)
    
    Returns:
        DataFrame with transaction data
    """
    if file_path is None:
        file_path = TRANSACTIONS_FILE
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"Transaction data file not found: {file_path}\n"
            f"Please run: python scripts/setup_data.py"
        )
    
    print(f"Loading transaction data: {file_path}")
    
    # Load data
    df = pd.read_csv(file_path)
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=42)
        print(f"   Sampled {sample_size} transactions")
    
    print(f"   Total: {len(df)} transactions")
    
    # Check if is_fraud column exists
    if 'is_fraud' in df.columns:
        print(f"   Normal: {(df['is_fraud']==0).sum()} transactions")
        print(f"   Fraud: {(df['is_fraud']==1).sum()} transactions ({(df['is_fraud']==1).sum()/len(df)*100:.3f}%)")
    else:
        print("   No fraud labels found in data")
    
    return df

def load_user_profiles(file_path=None):
    """
    Load user behavioral profiles
    Load user behavioral profiles
    
    Args:
        file_path: Path to JSON file (default: from config)
    
    Returns:
        Dictionary of user profiles
    """
    if file_path is None:
        file_path = USER_PROFILES_FILE
    
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(
            f"User profiles file not found: {file_path}\n"
            f"Please run: python scripts/setup_data.py"
        )
    
    print(f"Loading user profiles: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        profiles = json.load(f)
    
    print(f"   Total: {len(profiles)} users")
    
    return profiles

def validate_data(df):
    """
    Validate transaction data integrity
    Validate transaction data integrity
    """
    print("\nValidating data integrity...")
    
    required_columns = [
        'transaction_id', 'amount', 'timestamp', 'is_fraud',
        'is_cross_border', 'location_risk', 'behavioral_score',
        'transactions_last_hour', 'amount_last_24h'
    ]
    
    issues = []
    
    # Check required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    
    # Check for nulls only in existing columns
    existing_columns = [col for col in required_columns if col in df.columns]
    if existing_columns:
        null_counts = df[existing_columns].isnull().sum()
        if null_counts.any():
            issues.append(f"Null values found: {null_counts[null_counts > 0].to_dict()}")
    
    # Check value ranges for existing columns
    if 'amount' in df.columns and (df['amount'] < 0).any():
        issues.append("Negative amounts found")
    
    if 'is_fraud' in df.columns and not df['is_fraud'].isin([0, 1]).all():
        issues.append("is_fraud must be 0 or 1")
    
    if 'location_risk' in df.columns and not ((df['location_risk'] >= 0) & (df['location_risk'] <= 1)).all():
        issues.append("location_risk must be in [0,1] range")
    
    if 'behavioral_score' in df.columns and not ((df['behavioral_score'] >= 0) & (df['behavioral_score'] <= 1)).all():
        issues.append("behavioral_score must be in [0,1] range")
    
    if issues:
        print("Data validation failed:")
        for issue in issues:
            print(f"   - {issue}")
        return False
    else:
        print("Data validation passed")
        return True

def prepare_train_test_split(df, test_size=0.2, random_state=42):
    """
    Prepare train/test split maintaining fraud ratio
    Prepare train/test split maintaining fraud ratio
    """
    from sklearn.model_selection import train_test_split
    
    # Separate features and target
    feature_cols = [col for col in df.columns if col not in ['transaction_id', 'is_fraud', 'timestamp']]
    X = df[feature_cols]
    y = df['is_fraud']
    
    # Stratified split to maintain fraud ratio
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    print(f"\nData split:")
    print(f"   Training set: {len(X_train)} transactions (fraud: {y_train.sum()}, {y_train.sum()/len(y_train)*100:.3f}%)")
    print(f"   Test set: {len(X_test)} transactions (fraud: {y_test.sum()}, {y_test.sum()/len(y_test)*100:.3f}%)")
    
    return X_train, X_test, y_train, y_test

def get_data_summary(df):
    """
    Get summary statistics of transaction data
    Get summary statistics of transaction data
    """
    summary = {
        'total_transactions': len(df),
        'fraud_count': int(df['is_fraud'].sum()),
        'fraud_rate': float(df['is_fraud'].mean()),
        'amount_stats': {
            'mean': float(df['amount'].mean()),
            'median': float(df['amount'].median()),
            'std': float(df['amount'].std()),
            'min': float(df['amount'].min()),
            'max': float(df['amount'].max())
        },
        'cross_border_rate': float(df['is_cross_border'].mean()),
        'time_range': {
            'start': str(df['timestamp'].min()),
            'end': str(df['timestamp'].max())
        }
    }
    
    return summary
