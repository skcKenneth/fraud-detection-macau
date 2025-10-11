"""
Deepfake Detection for AI-Generated Content
深度偽造檢測：AI生成內容識別
"""
import numpy as np
from scipy.fft import fft

class DeepfakeDetector:
    """
    Detects AI-generated content (audio/video) in authentication
    檢測身份驗證中的AI生成內容（音頻/視頻）
    """
    
    def __init__(self):
        self.known_deepfake_signatures = set()
        self.detection_history = []
        
    def analyze_audio(self, audio_sample):
        """
        Analyze audio for AI-generated artifacts
        Simplified version using frequency domain features
        分析音頻中的AI生成偽影
        使用頻域特徵的簡化版本
        """
        deepfake_score = 0.0
        
        # Simulate frequency analysis using FFT
        frequencies = fft(audio_sample)
        power_spectrum = np.abs(frequencies[:len(frequencies)//2])
        
        if len(power_spectrum) < 1000:
            # Too short, unreliable
            return 0.5
        
        # Check for unnatural frequency patterns
        # AI voices often have unusual high-frequency content
        high_freq_ratio = np.sum(power_spectrum[1000:]) / (np.sum(power_spectrum) + 1e-6)
        
        if high_freq_ratio < 0.05 or high_freq_ratio > 0.35:
            deepfake_score += 0.35
        
        # Check for repetitive patterns (AI artifact)
        autocorr = np.correlate(audio_sample, audio_sample, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        if len(autocorr) > 100:
            peaks = self._find_peaks(autocorr[100:500])
            
            # Too many regular patterns indicate AI generation
            if len(peaks) > 10:
                deepfake_score += 0.30
        
        # Voice consistency check across segments
        segment_consistency = self._check_segment_consistency(audio_sample)
        if segment_consistency < 0.6:
            deepfake_score += 0.35
        
        return min(deepfake_score, 1.0)
    
    def analyze_video_frame(self, frame_data):
        """
        Analyze video frame for deepfake indicators
        Simplified using pixel-level analysis
        分析視頻幀的深度偽造指標
        使用像素級分析的簡化版本
        """
        deepfake_score = 0.0
        
        # Check for blending artifacts around face boundaries
        edge_consistency = self._check_edge_consistency(frame_data)
        if edge_consistency < 0.7:
            deepfake_score += 0.35
        
        # Check for unnatural eye blink patterns
        blink_pattern = self._analyze_blink_pattern(frame_data)
        if blink_pattern > 0.75:  # Too regular or too irregular
            deepfake_score += 0.35
        
        # Check for lighting inconsistencies
        lighting_score = self._check_lighting_consistency(frame_data)
        if lighting_score < 0.65:
            deepfake_score += 0.30
        
        return min(deepfake_score, 1.0)
    
    def _find_peaks(self, signal, threshold=0.5):
        """Find peaks in signal"""
        if len(signal) < 3:
            return []
        
        peaks = []
        max_val = np.max(np.abs(signal))
        
        for i in range(1, len(signal)-1):
            if (signal[i] > signal[i-1] and 
                signal[i] > signal[i+1] and 
                signal[i] > threshold * max_val):
                peaks.append(i)
        
        return peaks
    
    def _check_segment_consistency(self, audio_sample):
        """
        Check if audio segments have consistent characteristics
        檢查音頻段是否具有一致的特徵
        """
        # Divide into segments and compare
        segment_size = max(len(audio_sample) // 5, 1)
        segments = [audio_sample[i:i+segment_size] 
                   for i in range(0, len(audio_sample), segment_size)]
        
        # Calculate energy of each segment
        segment_energies = [np.mean(np.abs(seg)) for seg in segments if len(seg) == segment_size]
        
        if len(segment_energies) < 2:
            return 0.5
        
        # Consistency measured by coefficient of variation
        mean_energy = np.mean(segment_energies)
        std_energy = np.std(segment_energies)
        
        if mean_energy == 0:
            return 0.5
        
        cv = std_energy / mean_energy
        consistency = max(0, min(1, 1 - cv))
        
        return consistency
    
    def _check_edge_consistency(self, frame_data):
        """
        Check for blending artifacts (simulated)
        檢查混合偽影（模擬）
        """
        # Simulate edge detection analysis
        # In real implementation, would use actual computer vision
        edge_variance = np.random.normal(0.78, 0.12)
        return max(0, min(1, edge_variance))
    
    def _analyze_blink_pattern(self, frame_data):
        """
        Analyze eye blink naturalness (simulated)
        分析眼睛眨動自然度（模擬）
        """
        # Simulate blink pattern analysis
        # Real implementation would track eye movements
        blink_score = np.random.normal(0.35, 0.18)
        return max(0, min(1, blink_score))
    
    def _check_lighting_consistency(self, frame_data):
        """
        Check lighting consistency across face (simulated)
        檢查面部光照一致性（模擬）
        """
        # Simulate lighting analysis
        lighting_score = np.random.normal(0.75, 0.13)
        return max(0, min(1, lighting_score))
    
    def detect_synthetic_identity(self, audio_sample=None, video_frame=None, threshold=0.6):
        """
        Overall synthetic identity detection
        整體合成身份檢測
        """
        scores = []
        details = {}
        
        if audio_sample is not None:
            audio_score = self.analyze_audio(audio_sample)
            scores.append(audio_score)
            details['audio_score'] = float(audio_score)
        
        if video_frame is not None:
            video_score = self.analyze_video_frame(video_frame)
            scores.append(video_score)
            details['video_score'] = float(video_score)
        
        if not scores:
            return {
                'is_deepfake': False, 
                'confidence': 0.0,
                'overall_score': 0.0
            }
        
        overall_score = np.mean(scores)
        is_deepfake = overall_score >= threshold
        
        # Confidence increases with distance from threshold
        if is_deepfake:
            confidence = min((overall_score - threshold) / (1 - threshold), 1.0) * 0.8 + 0.2
        else:
            confidence = min((threshold - overall_score) / threshold, 1.0) * 0.8 + 0.2
        
        result = {
            'is_deepfake': bool(is_deepfake),
            'confidence': float(confidence),
            'overall_score': float(overall_score),
            'threshold': float(threshold),
            **details
        }
        
        # Store in history
        self.detection_history.append(result)
        
        return result
    
    def get_detection_stats(self):
        """Get statistics from detection history"""
        if not self.detection_history:
            return None
        
        total = len(self.detection_history)
        deepfakes = sum(1 for d in self.detection_history if d['is_deepfake'])
        
        return {
            'total_detections': total,
            'deepfake_count': deepfakes,
            'legitimate_count': total - deepfakes,
            'deepfake_rate': deepfakes / total if total > 0 else 0
        }
