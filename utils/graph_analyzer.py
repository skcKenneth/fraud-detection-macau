"""
Money Laundering Network Analysis using Graph Theory
基於圖論的洗錢網絡分析
"""
import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import (SUSPICIOUS_CIRCLE_LENGTH, RAPID_TRANSFER_WINDOW, 
                    MIN_LAYERING_HOPS, SMURFING_THRESHOLD, SMURFING_MIN_TRANSACTIONS)

class MoneyLaunderingDetector:
    """
    Uses graph analysis to detect money laundering patterns
    across cross-border networks
    使用圖分析檢測跨境網絡中的洗錢模式
    """
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.transactions = []
        self.accounts = set()
    
    def add_transaction(self, from_account, to_account, amount, timestamp, location=None):
        """
        Add transaction to network graph
        將交易添加到網絡圖
        """
        # Add to graph with edge attributes
        if self.graph.has_edge(from_account, to_account):
            # Update existing edge (add to amount)
            self.graph[from_account][to_account]['amount'] += amount
            self.graph[from_account][to_account]['count'] += 1
        else:
            self.graph.add_edge(
                from_account,
                to_account,
                amount=amount,
                timestamp=timestamp,
                location=location,
                count=1
            )
        
        # Store transaction details
        self.transactions.append({
            'from': from_account,
            'to': to_account,
            'amount': amount,
            'timestamp': timestamp,
            'location': location
        })
        
        # Track accounts
        self.accounts.add(from_account)
        self.accounts.add(to_account)
    
    def detect_circular_transactions(self, min_length=None):
        """
        Detect circular money flows indicating layering
        檢測表明分層的環形資金流動
        """
        if min_length is None:
            min_length = SUSPICIOUS_CIRCLE_LENGTH
        
        suspicious_circles = []
        
        try:
            # Find all simple cycles in the graph
            cycles = list(nx.simple_cycles(self.graph))
            
            for cycle in cycles:
                if len(cycle) >= min_length:
                    # Calculate total amount flowing in circle
                    cycle_amount = 0
                    cycle_edges = []
                    
                    for i in range(len(cycle)):
                        from_node = cycle[i]
                        to_node = cycle[(i+1) % len(cycle)]
                        
                        if self.graph.has_edge(from_node, to_node):
                            edge_data = self.graph[from_node][to_node]
                            cycle_amount += edge_data['amount']
                            cycle_edges.append({
                                'from': from_node,
                                'to': to_node,
                                'amount': edge_data['amount']
                            })
                    
                    # Calculate risk score based on cycle length and amount
                    risk_score = min(len(cycle) / 10 * 0.5 + 
                                   min(cycle_amount / 100000, 1.0) * 0.5, 1.0)
                    
                    suspicious_circles.append({
                        'accounts': cycle,
                        'length': len(cycle),
                        'total_amount': cycle_amount,
                        'edges': cycle_edges,
                        'risk_score': risk_score
                    })
            
            # Sort by risk score
            suspicious_circles.sort(key=lambda x: x['risk_score'], reverse=True)
            
        except Exception as e:
            print(f"循環檢測錯誤: {e}")
        
        return suspicious_circles
    
    def detect_rapid_layering(self, time_window=None, min_hops=None):
        """
        Detect rapid transfers through multiple accounts (layering)
        檢測通過多個帳戶的快速轉移（分層）
        """
        if time_window is None:
            time_window = RAPID_TRANSFER_WINDOW
        if min_hops is None:
            min_hops = MIN_LAYERING_HOPS
        
        df = pd.DataFrame(self.transactions)
        if df.empty:
            return []
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        suspicious_chains = []
        
        # Group by source account
        for account in df['from'].unique():
            account_txns = df[df['from'] == account].copy()
            
            if len(account_txns) < min_hops:
                continue
            
            # Check for rapid successive transactions
            for i in range(len(account_txns) - min_hops + 1):
                window = account_txns.iloc[i:i+min_hops]
                time_diff = (window['timestamp'].max() - window['timestamp'].min()).total_seconds()
                
                if time_diff <= time_window:
                    # Found rapid layering
                    total_amount = window['amount'].sum()
                    
                    # Calculate risk score
                    speed_factor = 1 - (time_diff / time_window)  # Faster = higher risk
                    amount_factor = min(total_amount / 50000, 1.0)
                    hops_factor = min(len(window) / 10, 1.0)
                    
                    risk_score = (speed_factor * 0.4 + amount_factor * 0.3 + hops_factor * 0.3)
                    
                    suspicious_chains.append({
                        'source_account': account,
                        'hops': len(window),
                        'time_window_seconds': time_diff,
                        'total_amount': total_amount,
                        'destinations': window['to'].tolist(),
                        'timestamps': window['timestamp'].tolist(),
                        'risk_score': risk_score
                    })
        
        # Sort by risk score
        suspicious_chains.sort(key=lambda x: x['risk_score'], reverse=True)
        
        return suspicious_chains
    
    def detect_smurfing(self, threshold=None, min_txns=None):
        """
        Detect structuring (smurfing) - many small transactions to avoid reporting
        檢測結構化（螞蟻搬家）- 許多小額交易以避免報告
        """
        if threshold is None:
            threshold = SMURFING_THRESHOLD
        if min_txns is None:
            min_txns = SMURFING_MIN_TRANSACTIONS
        
        df = pd.DataFrame(self.transactions)
        if df.empty:
            return []
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        suspicious_patterns = []
        
        # Group by account pairs
        grouped = df.groupby(['from', 'to'])
        
        for (from_acc, to_acc), group in grouped:
            # Check for multiple transactions
            if len(group) < min_txns:
                continue
            
            # Group by date
            group['date'] = group['timestamp'].dt.date
            daily_txns = group.groupby('date')
            
            for date, day_txns in daily_txns:
                if len(day_txns) >= min_txns:
                    total = day_txns['amount'].sum()
                    avg_amount = day_txns['amount'].mean()
                    max_amount = day_txns['amount'].max()
                    
                    # Suspicious if: multiple small txns totaling large amount
                    if total > threshold and avg_amount < threshold * 0.5:
                        # Calculate risk score
                        frequency_factor = min(len(day_txns) / 10, 1.0)
                        amount_factor = min(total / (threshold * 3), 1.0)
                        consistency_factor = 1 - (day_txns['amount'].std() / (avg_amount + 1))
                        
                        risk_score = (frequency_factor * 0.4 + 
                                    amount_factor * 0.3 + 
                                    consistency_factor * 0.3)
                        
                        suspicious_patterns.append({
                            'from_account': from_acc,
                            'to_account': to_acc,
                            'date': str(date),
                            'num_transactions': len(day_txns),
                            'total_amount': float(total),
                            'avg_amount': float(avg_amount),
                            'max_amount': float(max_amount),
                            'risk_score': float(risk_score)
                        })
        
        # Sort by risk score
        suspicious_patterns.sort(key=lambda x: x['risk_score'], reverse=True)
        
        return suspicious_patterns
    
    def get_network_risk_score(self, account):
        """
        Calculate overall risk score for an account based on network position
        基於網絡位置計算帳戶的整體風險分數
        """
        if account not in self.graph:
            return 0.0
        
        risk_factors = []
        
        # 1. High degree (connected to many accounts) - suspicious
        degree = self.graph.degree(account)
        degree_risk = min(degree / 20, 1.0)
        risk_factors.append(degree_risk * 0.3)
        
        # 2. Betweenness centrality (bridge between communities) - suspicious
        try:
            if len(self.graph) > 1:
                centrality = nx.betweenness_centrality(self.graph, weight='amount')
                account_centrality = centrality.get(account, 0)
                risk_factors.append(account_centrality * 0.4)
            else:
                risk_factors.append(0.0)
        except:
            risk_factors.append(0.0)
        
        # 3. Low clustering coefficient (part of loose network) - suspicious
        try:
            if len(self.graph) > 2:
                undirected = self.graph.to_undirected()
                clustering = nx.clustering(undirected)
                account_clustering = clustering.get(account, 0)
                # Low clustering = higher risk
                risk_factors.append((1 - account_clustering) * 0.3)
            else:
                risk_factors.append(0.0)
        except:
            risk_factors.append(0.0)
        
        overall_risk = sum(risk_factors)
        return min(overall_risk, 1.0)
    
    def get_network_statistics(self):
        """
        Get overall network statistics
        獲取整體網絡統計信息
        """
        if len(self.graph) == 0:
            return None
        
        stats = {
            'num_accounts': len(self.accounts),
            'num_transactions': len(self.transactions),
            'num_edges': self.graph.number_of_edges(),
            'total_amount': sum(t['amount'] for t in self.transactions),
            'avg_amount': np.mean([t['amount'] for t in self.transactions]),
            'density': nx.density(self.graph),
            'num_connected_components': nx.number_weakly_connected_components(self.graph)
        }
        
        return stats
    
    def get_top_risk_accounts(self, top_n=10):
        """
        Get accounts with highest risk scores
        獲取風險分數最高的帳戶
        """
        risk_scores = []
        
        for account in self.accounts:
            risk = self.get_network_risk_score(account)
            if risk > 0:
                # Get account activity
                out_degree = self.graph.out_degree(account)
                in_degree = self.graph.in_degree(account)
                
                # Calculate total amounts
                total_sent = sum(self.graph[account][neighbor]['amount'] 
                               for neighbor in self.graph.successors(account)
                               if self.graph.has_edge(account, neighbor))
                total_received = sum(self.graph[neighbor][account]['amount'] 
                                   for neighbor in self.graph.predecessors(account)
                                   if self.graph.has_edge(neighbor, account))
                
                risk_scores.append({
                    'account': account,
                    'risk_score': risk,
                    'out_degree': out_degree,
                    'in_degree': in_degree,
                    'total_sent': total_sent,
                    'total_received': total_received
                })
        
        # Sort by risk score
        risk_scores.sort(key=lambda x: x['risk_score'], reverse=True)
        
        return risk_scores[:top_n]
    
    def clear(self):
        """Clear all data"""
        self.graph.clear()
        self.transactions.clear()
        self.accounts.clear()
