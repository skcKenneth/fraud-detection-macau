"""
Ensemble Fraud Detection Model
Ensemble Fraud Detection Model
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from config import RANDOM_STATE, SMOTE_K_NEIGHBORS, RF_WEIGHT, GB_WEIGHT

# Try to import SMOTE, but make it optional for compatibility
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Warning: imbalanced-learn not available. SMOTE will be disabled. Using class_weight='balanced' instead.")

class EnsembleFraudDetector:
    """
    Ensemble model combining Random Forest and Gradient Boosting
    for handling imbalanced fraud detection
    Ensemble model combining Random Forest and Gradient Boosting for handling imbalanced fraud detection
    """
    
    def __init__(self):
        self.rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=RANDOM_STATE,
            n_jobs=-1
        )
        self.gb_model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=5,
            random_state=RANDOM_STATE
        )
        self.scaler = StandardScaler()
        if SMOTE_AVAILABLE:
            self.smote = SMOTE(random_state=RANDOM_STATE, k_neighbors=SMOTE_K_NEIGHBORS)
        else:
            self.smote = None
        self.is_trained = False
        self.feature_names = None
        
    def prepare_features(self, df):
        """Extract features from transaction data"""
        features = pd.DataFrame()
        
        # Amount-based features
        if 'amount' in df.columns and len(df) > 0:
            features['amount'] = df['amount'].fillna(0)
            features['log_amount'] = np.log1p(features['amount'])
            
            # Calculate z-score for amount
            if len(df) > 1:
                amount_mean = df['amount'].mean()
                amount_std = df['amount'].std()
                if pd.isna(amount_mean):
                    amount_mean = 0.0
                if pd.isna(amount_std) or amount_std == 0:
                    amount_std = 1.0
            else:
                amount_mean = df['amount'].iloc[0] if len(df) > 0 else 0.0
                amount_std = 1.0
            features['amount_zscore'] = (features['amount'] - amount_mean) / (amount_std + 1e-6)
        else:
            # Create dummy amount features
            features['amount'] = np.random.uniform(10, 1000, len(df))
            features['log_amount'] = np.log1p(features['amount'])
            features['amount_zscore'] = np.random.normal(0, 1, len(df))
        
        # Time-based features
        if 'timestamp' in df.columns:
            if 'hour' in df.columns:
                features['hour'] = df['hour']
            else:
                features['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            
            # Ensure we have arrays for boolean operations
            if len(df) == 1:
                hour_val = features['hour'].iloc[0]
                features['is_night'] = np.array([1 if (hour_val >= 23) or (hour_val <= 5) else 0])
                dayofweek_val = pd.to_datetime(df['timestamp']).dt.dayofweek.iloc[0]
                features['is_weekend'] = np.array([1 if dayofweek_val in [5, 6] else 0])
            else:
                features['is_night'] = ((features['hour'] >= 23) | (features['hour'] <= 5)).astype(int)
                features['is_weekend'] = pd.to_datetime(df['timestamp']).dt.dayofweek.isin([5, 6]).astype(int)
        else:
            # Create dummy time features
            features['hour'] = np.random.randint(0, 24, len(df))
            # Ensure we have arrays for boolean operations
            if len(df) == 1:
                features['is_night'] = np.array([1 if (features['hour'].iloc[0] >= 23) or (features['hour'].iloc[0] <= 5) else 0])
                features['is_weekend'] = np.random.choice([0, 1], 1, p=[0.7, 0.3])
            else:
                features['is_night'] = ((features['hour'] >= 23) | (features['hour'] <= 5)).astype(int)
                features['is_weekend'] = np.random.choice([0, 1], len(df), p=[0.7, 0.3])
        
        # Geographic features
        if 'is_cross_border' in df.columns:
            features['is_cross_border'] = df['is_cross_border'].astype(int)
        else:
            features['is_cross_border'] = np.random.choice([0, 1], len(df), p=[0.8, 0.2])
        
        if 'location_risk' in df.columns:
            features['location_risk'] = df['location_risk']
        else:
            features['location_risk'] = np.random.uniform(0, 1, len(df))
        
        # Behavioral features
        if 'behavioral_score' in df.columns:
            features['behavioral_score'] = df['behavioral_score']
        else:
            features['behavioral_score'] = np.random.uniform(0, 1, len(df))
        
        # Velocity features
        if 'transactions_last_hour' in df.columns:
            features['transactions_last_hour'] = df['transactions_last_hour']
        else:
            features['transactions_last_hour'] = np.random.poisson(2, len(df))
        
        if 'amount_last_24h' in df.columns:
            features['amount_last_24h'] = df['amount_last_24h']
            features['log_amount_24h'] = np.log1p(df['amount_last_24h'])
        else:
            features['amount_last_24h'] = np.random.uniform(100, 5000, len(df))
            features['log_amount_24h'] = np.log1p(features['amount_last_24h'])
        
        # Interaction features
        features['amount_x_cross_border'] = features['amount'] * features['is_cross_border']
        features['behavioral_x_location'] = features['behavioral_score'] * features['location_risk']
        
        self.feature_names = features.columns.tolist()
        
        return features
    
    def train(self, X, y):
        """Train ensemble model with SMOTE for imbalanced data"""
        print(f"\nTraining fraud detection model...")
        print(f"   Original data: {len(X)} transactions")
        print(f"   Fraud ratio: {y.sum()}/{len(y)} ({y.sum()/len(y)*100:.3f}%)")
        
        # Apply SMOTE to handle class imbalance (if available)
        if self.smote is not None:
            try:
                X_resampled, y_resampled = self.smote.fit_resample(X, y)
                print(f"   After SMOTE: {len(X_resampled)} transactions")
                print(f"   Fraud ratio: {y_resampled.sum()}/{len(y_resampled)} ({y_resampled.sum()/len(y_resampled)*100:.1f}%)")
            except Exception as e:
                print(f"   SMOTE failed ({e}), using original data")
                X_resampled, y_resampled = X, y
        else:
            # SMOTE not available, use original data (class_weight='balanced' already handles imbalance)
            print(f"   SMOTE not available, using original data with class_weight='balanced'")
            X_resampled, y_resampled = X, y
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_resampled)
        
        # Train both models
        print("   Training Random Forest...")
        self.rf_model.fit(X_scaled, y_resampled)
        
        print("   Training Gradient Boosting...")
        self.gb_model.fit(X_scaled, y_resampled)
        
        self.is_trained = True
        print("Model training completed!")
        
    def predict_proba(self, X):
        """Get fraud probability from ensemble"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        
        # Get predictions from both models
        rf_proba = self.rf_model.predict_proba(X_scaled)[:, 1]
        gb_proba = self.gb_model.predict_proba(X_scaled)[:, 1]
        
        # Weighted ensemble
        ensemble_proba = RF_WEIGHT * rf_proba + GB_WEIGHT * gb_proba
        
        return ensemble_proba
    
    def predict(self, X, threshold=0.75):
        """Predict fraud with custom threshold"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)
    
    def get_feature_importance(self):
        """Get feature importance from Random Forest"""
        if not self.is_trained:
            return None
        
        importance = self.rf_model.feature_importances_
        
        if self.feature_names:
            return dict(zip(self.feature_names, importance))
        else:
            return importance
    
    def evaluate(self, X, y, threshold=0.75):
        """Evaluate model performance"""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Get predictions
        y_proba = self.predict_proba(X)
        y_pred = (y_proba >= threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'auc': roc_auc_score(y, y_proba)
        }
        
        return metrics
