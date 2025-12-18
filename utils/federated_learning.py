"""
Federated Learning for Cross-Border Fraud Detection
跨境欺詐檢測的聯邦學習
"""
import numpy as np
from models.fraud_detector import EnsembleFraudDetector
from config import FL_ROUNDS, FL_LOCAL_WEIGHT, FL_GLOBAL_WEIGHT

class FederatedLearning:
    """
    Simulates federated learning across Macau, Hong Kong, and Zhuhai banks
    模擬澳門、香港和珠海銀行間的聯邦學習
    """
    
    def __init__(self, bank_names):
        self.bank_names = bank_names
        self.local_models = {bank: EnsembleFraudDetector() for bank in bank_names}
        self.global_model = EnsembleFraudDetector()
        self.training_rounds = 0
        self.round_history = []
    
    def train_local_models(self, bank_data):
        """
        Each bank trains on local data
        每個銀行在本地數據上訓練
        
        Args:
            bank_data: dict with bank_name as key, (X, y) tuple as value
        """
        print(f"\n{'='*60}")
        print(f"聯邦學習 - 第 {self.training_rounds + 1} 輪")
        print(f"{'='*60}")
        
        round_info = {
            'round': self.training_rounds + 1,
            'banks': []
        }
        
        for bank, (X, y) in bank_data.items():
            if bank not in self.bank_names:
                print(f"⚠️  {bank} 不在參與銀行列表中")
                continue
            
            print(f"\n📍 {bank}")
            print(f"   訓練數據: {len(X)} 筆交易")
            print(f"   欺詐比例: {y.sum()}/{len(y)} ({y.sum()/len(y)*100:.3f}%)")
            
            # Train local model
            self.local_models[bank].train(X, y)
            
            # Record training info
            round_info['banks'].append({
                'name': bank,
                'samples': len(X),
                'fraud_count': int(y.sum()),
                'fraud_rate': float(y.sum() / len(y))
            })
        
        self.round_history.append(round_info)
    
    def aggregate_models(self):
        """
        Aggregate local models into global model (simplified averaging)
        將本地模型聚合為全局模型（簡化平均）
        """
        print(f"\n🔗 聚合模型...")
        
        # Collect feature importances from all trained models
        all_importances = []
        participating_banks = []
        
        for bank, model in self.local_models.items():
            if model.is_trained:
                importance = model.get_feature_importance()
                if importance is not None:
                    if isinstance(importance, dict):
                        importance = list(importance.values())
                    # Ensure importance is a list/array
                    if not isinstance(importance, (list, np.ndarray)):
                        continue
                    all_importances.append(importance)
                    participating_banks.append(bank)
        
        if all_importances:
            # Ensure all importances have the same length
            lengths = [len(imp) for imp in all_importances]
            if len(set(lengths)) > 1:
                # Use the minimum length to avoid errors
                min_len = min(lengths)
                all_importances = [imp[:min_len] if len(imp) > min_len else imp for imp in all_importances]
            # Average feature importances
            avg_importance = np.mean(all_importances, axis=0)
            
            print(f"✓ 全局模型已更新")
            print(f"   參與銀行: {len(participating_banks)}/{len(self.bank_names)}")
            print(f"   參與機構: {', '.join(participating_banks)}")
        else:
            print("⚠️  沒有訓練好的模型可聚合")
            avg_importance = None
        
        self.training_rounds += 1
        
        return {
            'round': self.training_rounds,
            'num_banks_participated': len(participating_banks),
            'participating_banks': participating_banks,
            'global_feature_importance': avg_importance.tolist() if avg_importance is not None else None
        }
    
    def predict_with_global_model(self, X, bank_name):
        """
        Use global model enhanced with local bank knowledge
        使用增強本地銀行知識的全局模型
        """
        if bank_name not in self.local_models:
            raise ValueError(f"{bank_name} 不在參與銀行列表中")
        
        if not self.local_models[bank_name].is_trained:
            raise ValueError(f"{bank_name} 模型尚未訓練")
        
        # Get local prediction
        local_proba = self.local_models[bank_name].predict_proba(X)
        
        # Get predictions from other banks (global knowledge)
        other_banks_proba = []
        for bank, model in self.local_models.items():
            if bank != bank_name and model.is_trained:
                try:
                    proba = model.predict_proba(X)
                    other_banks_proba.append(proba)
                except:
                    continue
        
        # Combine local and global predictions
        if other_banks_proba:
            global_contribution = np.mean(other_banks_proba, axis=0)
            # Weighted combination: more weight on local model
            combined_proba = (FL_LOCAL_WEIGHT * local_proba + 
                            FL_GLOBAL_WEIGHT * global_contribution)
        else:
            # If no other banks available, use only local
            combined_proba = local_proba
        
        return combined_proba
    
    def evaluate_all_banks(self, bank_test_data, threshold=0.75):
        """
        Evaluate all banks on their test data
        在測試數據上評估所有銀行
        
        Args:
            bank_test_data: dict with bank_name as key, (X_test, y_test) tuple as value
        """
        results = {}
        
        for bank, (X_test, y_test) in bank_test_data.items():
            if bank not in self.local_models or not self.local_models[bank].is_trained:
                continue
            
            # Evaluate with local model only
            local_metrics = self.local_models[bank].evaluate(X_test, y_test, threshold)
            
            # Evaluate with federated model
            try:
                fed_proba = self.predict_with_global_model(X_test, bank)
                fed_pred = (fed_proba >= threshold).astype(int)
                
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
                
                fed_metrics = {
                    'accuracy': accuracy_score(y_test, fed_pred),
                    'precision': precision_score(y_test, fed_pred, zero_division=0),
                    'recall': recall_score(y_test, fed_pred, zero_division=0),
                    'f1': f1_score(y_test, fed_pred, zero_division=0),
                    'auc': roc_auc_score(y_test, fed_proba)
                }
            except:
                fed_metrics = local_metrics.copy()
            
            results[bank] = {
                'local': local_metrics,
                'federated': fed_metrics,
                'improvement': {
                    'accuracy': fed_metrics['accuracy'] - local_metrics['accuracy'],
                    'f1': fed_metrics['f1'] - local_metrics['f1'],
                    'auc': fed_metrics['auc'] - local_metrics['auc']
                }
            }
        
        return results
    
    def get_training_summary(self):
        """
        Get summary of federated learning process
        獲取聯邦學習過程的摘要
        """
        summary = {
            'total_rounds': self.training_rounds,
            'participating_banks': self.bank_names,
            'num_banks': len(self.bank_names),
            'models_trained': sum(1 for model in self.local_models.values() if model.is_trained),
            'round_history': self.round_history
        }
        return summary
    
    def get_performance_comparison(self):
        """
        Get performance comparison between local and federated models
        獲取本地模型和聯邦模型之間的性能比較
        """
        if not any(model.is_trained for model in self.local_models.values()):
            return None
        
        comparison = []
        
        for bank, model in self.local_models.items():
            if model.is_trained:
                comparison.append({
                    'bank': bank,
                    'is_trained': True
                })
        
        return comparison
