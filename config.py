"""
Configuration settings for Cross-Border Fraud Detection System
跨境支付欺詐檢測系統配置文件
"""
import numpy as np
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'

# Data files
TRANSACTIONS_FILE = DATA_DIR / 'sample_transactions.csv'
USER_PROFILES_FILE = DATA_DIR / 'user_profiles.json'

# Kaggle dataset
KAGGLE_DATASET = 'mlg-ulb/creditcardfraud'
KAGGLE_DATA_FILE = 'creditcard.csv'

# Simulation parameters
BANKS = ['澳門銀行 (Macau Bank)', '香港銀行 (Hong Kong Bank)', '珠海銀行 (Zhuhai Bank)']
BANK_CODES = {'MO': '澳門', 'HK': '香港', 'ZH': '珠海'}

# Fraud detection thresholds
FRAUD_THRESHOLD = 0.75  # Probability threshold for fraud alert (75%)
BEHAVIORAL_THRESHOLD = 0.65  # Behavioral anomaly threshold (65%)
NETWORK_RISK_THRESHOLD = 0.70  # Network risk threshold (70%)

# Feature weights for ensemble model
FEATURE_WEIGHTS = {
    'transaction_amount': 0.25,
    'cross_border': 0.20,
    'behavioral_score': 0.30,
    'network_risk': 0.15,
    'deepfake_risk': 0.10
}

# Behavioral biometrics parameters (in milliseconds/pixels)
NORMAL_KEYSTROKE_MEAN = 150  # Average keystroke interval (ms)
NORMAL_KEYSTROKE_STD = 30    # Standard deviation
NORMAL_MOUSE_VELOCITY = 500  # Average mouse velocity (px/s)
MOUSE_VELOCITY_STD = 100     # Standard deviation
NORMAL_SESSION_DURATION = 600  # Average session duration (seconds)

# Network analysis thresholds
SUSPICIOUS_CIRCLE_LENGTH = 4  # Minimum length for suspicious circular transactions
RAPID_TRANSFER_WINDOW = 3600  # Time window for rapid transfers (seconds = 1 hour)
MIN_LAYERING_HOPS = 3  # Minimum number of hops for layering detection
SMURFING_THRESHOLD = 10000  # Amount threshold for smurfing detection (MOP)
SMURFING_MIN_TRANSACTIONS = 3  # Minimum transactions per day to flag smurfing

# Model parameters
RANDOM_STATE = 42  # For reproducibility
SMOTE_K_NEIGHBORS = 3  # SMOTE k-neighbors (small due to low fraud rate)
TEST_SIZE = 0.2  # Train/test split ratio

# Ensemble model weights
RF_WEIGHT = 0.6  # Random Forest weight in ensemble
GB_WEIGHT = 0.4  # Gradient Boosting weight in ensemble

# Federated learning parameters
FL_ROUNDS = 5  # Number of federated learning rounds
FL_LOCAL_WEIGHT = 0.7  # Weight for local model predictions
FL_GLOBAL_WEIGHT = 0.3  # Weight for global model predictions

# Real-time processing
MAX_PROCESSING_TIME_MS = 100  # Maximum processing time per transaction (ms)
BATCH_SIZE = 32  # Batch size for processing

# Display settings
RECENT_TRANSACTIONS_COUNT = 20  # Number of recent transactions to display
CHART_HEIGHT = 400  # Default chart height (pixels)
TABLE_HEIGHT = 400  # Default table height (pixels)

# Alert thresholds for dashboard metrics
HIGH_FRAUD_ALERT = 50  # Alert if daily fraud count exceeds this
LOW_ACCURACY_ALERT = 0.95  # Alert if accuracy drops below this (95%)

# Currency
CURRENCY = 'MOP'  # Macau Pataca
CURRENCY_SYMBOL = 'MOP'

# Timezone
TIMEZONE = 'Asia/Macau'

# Logging
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
