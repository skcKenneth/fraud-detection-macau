"""
Download and process Kaggle Credit Card Fraud Detection dataset
Download and process Kaggle Credit Card Fraud Detection dataset
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import sys
import os

def download_from_kaggle(dataset_name='mlg-ulb/creditcardfraud'):
    """
    Download dataset from Kaggle using API
    Requires Kaggle API authentication
    """
    try:
        import opendatasets as od
        print("Downloading dataset from Kaggle...")
        print("   (First time requires Kaggle API credentials)")
        print("   If you don't have them, visit: https://www.kaggle.com/account")
        
        # Download to data directory
        data_dir = Path(__file__).parent
        download_dir = data_dir / 'creditcardfraud'
        
        if not download_dir.exists():
            od.download(f'https://www.kaggle.com/datasets/{dataset_name}', str(data_dir))
        
        # Check if downloaded
        csv_file = download_dir / 'creditcard.csv'
        if csv_file.exists():
            print(f"Download successful: {csv_file}")
            return csv_file
        else:
            print("Download failed: creditcard.csv not found")
            return None
            
    except ImportError:
        print("opendatasets not installed, please run: pip install opendatasets")
        return None
    except Exception as e:
        print(f"Download failed: {e}")
        print("   Please try manual download:")
        print("   1. Visit https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("   2. Download creditcard.csv")
        print(f"   3. Place in {Path(__file__).parent} folder")
        return None

def generate_sample_data():
    """
    Generate sample transaction data when Kaggle data is not available
    """
    print("Generating sample transaction data...")
    
    np.random.seed(42)
    n_transactions = 10000
    
    # Generate timestamps (last 30 days)
    start_time = datetime.now() - timedelta(days=30)
    timestamps = [start_time + timedelta(seconds=np.random.randint(0, 30*24*3600)) for _ in range(n_transactions)]
    
    # Generate amounts (log-normal distribution)
    amounts = np.random.lognormal(mean=6, sigma=1.5, size=n_transactions)
    amounts = np.clip(amounts, 1, 50000)  # Cap at 50,000 MOP
    
    # Generate fraud labels (2% fraud rate)
    fraud_labels = np.random.choice([0, 1], size=n_transactions, p=[0.98, 0.02])
    
    # Generate cross-border transactions (30% cross-border)
    cross_border = np.random.choice([0, 1], size=n_transactions, p=[0.7, 0.3])
    
    # Generate location risk scores
    location_risk = np.random.beta(2, 5, size=n_transactions)  # Skewed towards lower risk
    
    # Generate behavioral scores
    behavioral_score = np.random.beta(3, 3, size=n_transactions)  # More balanced
    
    # Generate velocity features
    transactions_last_hour = np.random.poisson(2, size=n_transactions)
    amount_last_24h = amounts * np.random.uniform(0.5, 3.0, size=n_transactions)
    
    # Generate account numbers
    from_accounts = [f'ACC_{i%1000:03d}' for i in range(n_transactions)]
    to_accounts = [f'ACC_{i%1000:03d}' for i in range(n_transactions)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'transaction_id': [f'TXN_{i:06d}' for i in range(n_transactions)],
        'from_account': from_accounts,
        'to_account': to_accounts,
        'amount': amounts,
        'timestamp': timestamps,
        'is_fraud': fraud_labels,
        'is_cross_border': cross_border,
        'location_risk': location_risk,
        'behavioral_score': behavioral_score,
        'transactions_last_hour': transactions_last_hour,
        'amount_last_24h': amount_last_24h
    })
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    print(f"Generated {len(df)} sample transactions")
    print(f"   Normal: {(df['is_fraud']==0).sum()} ({(df['is_fraud']==0).sum()/len(df)*100:.2f}%)")
    print(f"   Fraud: {(df['is_fraud']==1).sum()} ({(df['is_fraud']==1).sum()/len(df)*100:.3f}%)")
    print(f"   Cross-border: {df['is_cross_border'].sum()} ({df['is_cross_border'].sum()/len(df)*100:.2f}%)")
    
    return df

def process_kaggle_data(source_file):
    """
    Process Kaggle dataset into our format
    Process Kaggle dataset into our format
    """
    print(f"\nProcessing data: {source_file}")
    
    # Read original dataset
    df = pd.read_csv(source_file)
    print(f"   Original data: {len(df)} transactions")
    print(f"   Fraud transactions: {df['Class'].sum()} ({df['Class'].sum()/len(df)*100:.3f}%)")
    
    # Create timestamps (original data has time in seconds)
    base_time = datetime(2013, 9, 1)  # Dataset from Sept 2013
    df['timestamp'] = df['Time'].apply(lambda x: base_time + timedelta(seconds=x))
    
    # Extract hour from timestamp
    df['hour'] = (df['Time'] // 3600) % 24
    df['hour'] = df['hour'].astype(int)
    
    # Generate account IDs with regional prefixes (MO, HK, ZH)
    regions = ['MO', 'HK', 'ZH']
    n_accounts = 1000
    accounts = []
    for region in regions:
        accounts.extend([f"{region}{i:06d}" for i in range(1, n_accounts//3 + 1)])
    
    # Randomly assign accounts
    np.random.seed(42)
    df['from_account'] = np.random.choice(accounts, len(df))
    df['to_account'] = [np.random.choice([a for a in accounts if a != from_acc]) 
                        for from_acc in df['from_account']]
    
    # Determine if cross-border based on account prefixes
    df['is_cross_border'] = (df['from_account'].str[:2] != df['to_account'].str[:2]).astype(int)
    
    # Create derived features
    # Location risk (higher for larger amounts and fraud cases)
    df['location_risk'] = (df['Amount'] / df['Amount'].max() * 0.5 + 
                           df['Class'] * 0.4 + 
                           np.random.uniform(0, 0.1, len(df)))
    df['location_risk'] = df['location_risk'].clip(0, 1)
    
    # Behavioral score (anomalous for fraud)
    df['behavioral_score'] = np.where(
        df['Class'] == 1,
        np.random.uniform(0.5, 0.9, len(df)),  # High for fraud
        np.random.uniform(0.05, 0.35, len(df))  # Low for normal
    )
    
    # Transaction velocity features
    df['transactions_last_hour'] = np.where(
        df['Class'] == 1,
        np.random.poisson(4, len(df)),  # More frequent for fraud
        np.random.poisson(1.5, len(df))
    )
    
    # Amount in last 24h (cumulative)
    df['amount_last_24h'] = df.groupby('from_account')['Amount'].transform(
        lambda x: x.rolling(window=100, min_periods=1).sum()
    )
    
    # Merchant categories
    categories = ['retail', 'food', 'transport', 'entertainment', 'utilities', 'online', 'overseas']
    df['merchant_category'] = np.where(
        df['Class'] == 1,
        np.random.choice(['online', 'overseas', 'gambling'], len(df)),
        np.random.choice(categories[:5], len(df))
    )
    
    # Device type
    devices = ['mobile', 'web', 'atm']
    df['device_type'] = np.random.choice(devices, len(df))
    
    # Create final dataframe with our schema
    df_processed = pd.DataFrame({
        'transaction_id': [f"TXN{i:08d}" for i in range(len(df))],
        'from_account': df['from_account'],
        'to_account': df['to_account'],
        'amount': df['Amount'],
        'timestamp': df['timestamp'],
        'hour': df['hour'],
        'is_cross_border': df['is_cross_border'],
        'location_risk': df['location_risk'],
        'behavioral_score': df['behavioral_score'],
        'transactions_last_hour': df['transactions_last_hour'],
        'amount_last_24h': df['amount_last_24h'],
        'merchant_category': df['merchant_category'],
        'device_type': df['device_type'],
        'is_fraud': df['Class']
    })
    
    print(f"\nProcessing completed:")
    print(f"   Total transactions: {len(df_processed)}")
    print(f"   Normal transactions: {(df_processed['is_fraud']==0).sum()} ({(df_processed['is_fraud']==0).sum()/len(df_processed)*100:.2f}%)")
    print(f"   Fraud transactions: {(df_processed['is_fraud']==1).sum()} ({(df_processed['is_fraud']==1).sum()/len(df_processed)*100:.3f}%)")
    print(f"   Cross-border transactions: {df_processed['is_cross_border'].sum()} ({df_processed['is_cross_border'].sum()/len(df_processed)*100:.2f}%)")
    
    return df_processed

def setup_kaggle_data(output_file=None, force_download=False):
    """
    Main function to setup Kaggle data
    Main function: Setup Kaggle data
    """
    if output_file is None:
        output_file = Path(__file__).parent / 'sample_transactions.csv'
    
    # Check if already processed
    if output_file.exists() and not force_download:
        print(f"Data file already exists: {output_file}")
        df = pd.read_csv(output_file)
        print(f"   Contains {len(df)} transaction records")
        return output_file
    
    # Try to find existing downloaded file
    data_dir = Path(__file__).parent
    kaggle_dir = data_dir / 'creditcardfraud'
    kaggle_file = kaggle_dir / 'creditcard.csv'
    
    # If not exists, try to download
    if not kaggle_file.exists():
        print("Kaggle data file not found, attempting download...")
        downloaded_file = download_from_kaggle()
        if downloaded_file:
            kaggle_file = downloaded_file
        else:
            print("\nUnable to automatically download Kaggle data")
            print("   Generating sample data instead...")
            df = generate_sample_data()
            # Save processed data
            output_file.parent.mkdir(exist_ok=True)
            df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"Saved to: {output_file}")
            return output_file
    
    # Process the data
    df = process_kaggle_data(kaggle_file)
    
    # Save processed data
    output_file.parent.mkdir(exist_ok=True)
    df.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"\nSaved to: {output_file}")
    
    return output_file

if __name__ == "__main__":
    setup_kaggle_data(force_download=True)
