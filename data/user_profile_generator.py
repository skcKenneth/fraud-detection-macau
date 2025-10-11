"""
Generate realistic user behavioral profiles
Generate realistic user behavior profiles
"""
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy types"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def generate_user_profiles(n_users=100, seed=42):
    """
    Generate behavioral biometric profiles for users
    Generate user behavioral biometric profiles
    """
    np.random.seed(seed)
    
    print(f"\nGenerating user behavior profiles...")
    
    profiles = {}
    
    # User types with different behavioral patterns
    user_types = [
        ('conservative', 0.3),   # 30% - Slow, careful users
        ('moderate', 0.5),       # 50% - Average users
        ('active', 0.2)          # 20% - Fast, experienced users
    ]
    
    # First names for Macau (Chinese + Portuguese)
    first_names = [
        'Mingxuan', 'Siting', 'Weiqiang', 'Xiaofang', 'Jianguo', 'Lihua', 'Zhiqiang', 'Meiling',
        'João', 'Maria', 'Pedro', 'Ana', 'Carlos', 'Sofia', 'Miguel', 'Isabel',
        'Jiahao', 'Yating', 'Junjie', 'Shufen', 'Antonio', 'Catarina', 'Zixuan', 'Xinyi'
    ]
    
    # Last names
    last_names = ['Chen', 'Huang', 'Li', 'Lin', 'Zhang', 'Silva', 'Santos', 'Oliveira', 'Wang', 'Liu', 'Ferreira', 'Wu']
    
    for i in range(n_users):
        user_id = f"USER{i:04d}"
        
        # Select user type
        rand = np.random.rand()
        cumsum = 0
        selected_type = 'moderate'
        for utype, prob in user_types:
            cumsum += prob
            if rand <= cumsum:
                selected_type = utype
                break
        
        # Set baseline parameters based on user type
        if selected_type == 'conservative':
            keystroke_base = 180  # Slower typing
            mouse_base = 400      # Slower mouse
            session_base = 800    # Longer sessions
            transactions_per_month = np.random.poisson(5)
        elif selected_type == 'moderate':
            keystroke_base = 150
            mouse_base = 500
            session_base = 600
            transactions_per_month = np.random.poisson(12)
        else:  # active
            keystroke_base = 110  # Fast typing
            mouse_base = 680      # Fast mouse
            session_base = 400    # Shorter sessions
            transactions_per_month = np.random.poisson(25)
        
        # Add individual variation
        keystroke_mean = keystroke_base + np.random.uniform(-25, 25)
        keystroke_std = np.random.uniform(20, 45)
        mouse_velocity_mean = mouse_base + np.random.uniform(-100, 100)
        mouse_velocity_std = np.random.uniform(70, 130)
        avg_session_duration = session_base + np.random.uniform(-150, 150)
        
        # Typical login hours (different patterns)
        if selected_type == 'conservative':
            # Daytime hours
            typical_hours = list(np.random.choice(range(9, 18), size=np.random.randint(4, 7), replace=False))
        elif selected_type == 'moderate':
            # Mix of day and evening
            typical_hours = list(np.random.choice(range(8, 23), size=np.random.randint(5, 10), replace=False))
        else:  # active
            # Any time, including night
            typical_hours = list(np.random.choice(range(24), size=np.random.randint(8, 15), replace=False))
        
        # Account age
        account_age_days = int(np.random.uniform(90, 1825))  # 3 months to 5 years
        registration_date = datetime.now() - timedelta(days=account_age_days)
        
        # Generate name
        name = f"{np.random.choice(last_names)}{np.random.choice(first_names)}"
        
        # Preferred device
        device_prefs = ['mobile', 'web', 'both']
        device_weights = [0.5, 0.3, 0.2]
        preferred_device = np.random.choice(device_prefs, p=device_weights)
        
        # Risk score (mostly low, some medium, few high)
        risk_categories = [
            (0.05, 0.25, 0.7),   # Low risk (70%)
            (0.25, 0.45, 0.25),  # Medium risk (25%)
            (0.45, 0.80, 0.05)   # High risk (5%)
        ]
        category = np.random.choice(len(risk_categories), p=[0.7, 0.25, 0.05])
        risk_min, risk_max, _ = risk_categories[category]
        risk_score = np.random.uniform(risk_min, risk_max)
        
        # Create profile
        profiles[user_id] = {
            'user_id': user_id,
            'name': name,
            'user_type': selected_type,
            'keystroke_mean': float(np.round(keystroke_mean, 2)),
            'keystroke_std': float(np.round(keystroke_std, 2)),
            'mouse_velocity_mean': float(np.round(mouse_velocity_mean, 2)),
            'mouse_velocity_std': float(np.round(mouse_velocity_std, 2)),
            'avg_session_duration': float(np.round(avg_session_duration, 2)),
            'typical_login_hours': sorted(typical_hours),
            'preferred_device': preferred_device,
            'account_age_days': int(account_age_days),
            'registration_date': registration_date.strftime('%Y-%m-%d'),
            'avg_monthly_transactions': int(transactions_per_month),
            'risk_score': float(np.round(risk_score, 3)),
            'account_status': 'active' if risk_score < 0.6 else 'monitoring'
        }
    
    print(f"Generated {len(profiles)} user profiles")
    print(f"   Conservative users: {sum(1 for p in profiles.values() if p['user_type']=='conservative')}")
    print(f"   Moderate users: {sum(1 for p in profiles.values() if p['user_type']=='moderate')}")
    print(f"   Active users: {sum(1 for p in profiles.values() if p['user_type']=='active')}")
    
    return profiles

def setup_user_profiles(output_file=None, n_users=100, force_regenerate=False):
    """
    Main function to setup user profiles
    Main function: Setup user profiles
    """
    if output_file is None:
        output_file = Path(__file__).parent / 'user_profiles.json'
    
    # Check if already exists
    if output_file.exists() and not force_regenerate:
        print(f"User profiles already exist: {output_file}")
        with open(output_file, 'r', encoding='utf-8') as f:
            profiles = json.load(f)
        print(f"   Contains {len(profiles)} users")
        return output_file
    
    # Generate profiles
    profiles = generate_user_profiles(n_users)
    
    # Save to file
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(profiles, f, indent=2, ensure_ascii=False, cls=NumpyEncoder)
    
    print(f"Saved to: {output_file}")
    
    return output_file

if __name__ == "__main__":
    setup_user_profiles(n_users=100, force_regenerate=True)
