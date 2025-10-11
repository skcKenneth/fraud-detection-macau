"""
Behavioral Biometrics Analysis
行為生物識別分析
"""
import numpy as np
from scipy import stats
from config import (NORMAL_KEYSTROKE_MEAN, NORMAL_KEYSTROKE_STD, 
                    NORMAL_MOUSE_VELOCITY, MOUSE_VELOCITY_STD,
                    BEHAVIORAL_THRESHOLD)

class BehavioralBiometrics:
    """
    Analyzes user interaction patterns to detect account takeover
    分析用戶交互模式以檢測帳戶盜用
    """
    
    def __init__(self):
        self.user_profiles = {}
    
    def create_user_profile(self, user_id, keystroke_data, mouse_data, session_data):
        """
        Create behavioral profile for legitimate user
        為合法用戶創建行為檔案
        """
        profile = {
            'keystroke_mean': float(np.mean(keystroke_data)),
            'keystroke_std': float(np.std(keystroke_data)),
            'mouse_velocity_mean': float(np.mean(mouse_data)),
            'mouse_velocity_std': float(np.std(mouse_data)),
            'avg_session_duration': float(np.mean(session_data)),
            'typing_rhythm': self._calculate_rhythm(keystroke_data),
            'sample_size': len(keystroke_data)
        }
        self.user_profiles[user_id] = profile
        return profile
    
    def load_profile(self, user_id, profile_dict):
        """
        Load existing user profile from dictionary
        從字典加載現有用戶檔案
        """
        self.user_profiles[user_id] = profile_dict
    
    def _calculate_rhythm(self, keystroke_data):
        """
        Calculate typing rhythm pattern (coefficient of variation)
        計算打字節奏模式（變異係數）
        """
        if len(keystroke_data) < 2:
            return 0.0
        intervals = np.diff(keystroke_data)
        mean_interval = np.mean(intervals)
        if mean_interval == 0:
            return 0.0
        return float(np.std(intervals) / mean_interval)
    
    def analyze_session(self, user_id, current_keystroke, current_mouse, current_duration):
        """
        Analyze current session against user profile
        Returns anomaly score (0-1, higher = more suspicious)
        根據用戶檔案分析當前會話
        返回異常分數（0-1，越高越可疑）
        """
        if user_id not in self.user_profiles:
            # New user - neutral score
            return 0.5
        
        profile = self.user_profiles[user_id]
        anomaly_scores = []
        
        # Keystroke timing anomaly (z-score based)
        keystroke_z = abs(current_keystroke - profile['keystroke_mean']) / (profile['keystroke_std'] + 1e-6)
        keystroke_anomaly = min(keystroke_z / 3, 1.0)  # Normalize to 0-1 (3 sigma rule)
        anomaly_scores.append(keystroke_anomaly)
        
        # Mouse velocity anomaly
        mouse_z = abs(current_mouse - profile['mouse_velocity_mean']) / (profile['mouse_velocity_std'] + 1e-6)
        mouse_anomaly = min(mouse_z / 3, 1.0)
        anomaly_scores.append(mouse_anomaly)
        
        # Session duration anomaly
        duration_z = abs(current_duration - profile['avg_session_duration']) / (profile['avg_session_duration'] * 0.5 + 1e-6)
        duration_anomaly = min(duration_z / 2, 1.0)
        anomaly_scores.append(duration_anomaly)
        
        # Combined anomaly score (weighted average)
        # Keystroke is most reliable, then mouse, then duration
        weights = [0.45, 0.35, 0.20]
        overall_anomaly = np.average(anomaly_scores, weights=weights)
        
        return float(overall_anomaly)
    
    def detect_account_takeover(self, user_id, current_data, threshold=None):
        """
        Detect if current session likely represents account takeover
        檢測當前會話是否可能代表帳戶盜用
        """
        if threshold is None:
            threshold = BEHAVIORAL_THRESHOLD
        
        anomaly_score = self.analyze_session(
            user_id,
            current_data['keystroke'],
            current_data['mouse_velocity'],
            current_data['session_duration']
        )
        
        is_takeover = anomaly_score >= threshold
        
        # Confidence is higher when score is far from threshold
        if is_takeover:
            confidence = min(anomaly_score * 1.2, 1.0)
        else:
            confidence = max(1.0 - anomaly_score * 1.2, 0.0)
        
        return {
            'is_suspicious': bool(is_takeover),
            'anomaly_score': float(anomaly_score),
            'confidence': float(confidence),
            'threshold': float(threshold)
        }
    
    def update_profile(self, user_id, new_keystroke, new_mouse, new_duration):
        """
        Update user profile with new legitimate session data (incremental learning)
        用新的合法會話數據更新用戶檔案（增量學習）
        """
        if user_id not in self.user_profiles:
            return False
        
        profile = self.user_profiles[user_id]
        n = profile.get('sample_size', 10)
        
        # Exponential moving average for adaptation
        alpha = 0.1  # Learning rate
        
        profile['keystroke_mean'] = (1-alpha) * profile['keystroke_mean'] + alpha * new_keystroke
        profile['mouse_velocity_mean'] = (1-alpha) * profile['mouse_velocity_mean'] + alpha * new_mouse
        profile['avg_session_duration'] = (1-alpha) * profile['avg_session_duration'] + alpha * new_duration
        
        return True
    
    def get_profile_summary(self, user_id):
        """Get summary of user profile"""
        if user_id not in self.user_profiles:
            return None
        
        return self.user_profiles[user_id]
