# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from typing import List, Dict
from ui_utils import console, Table, Panel

class FeatureAnalyzer:
    """Advanced scanner to evaluate feature health and relevance"""
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        
    def get_feature_relevance(self, target_col: str = 'direction_label') -> pd.DataFrame:
        """
        Calculates Spearman Correlation for all features against the target.
        Spearman is better for capture non-linear market patterns.
        """
        if target_col not in self.df.columns:
            return pd.DataFrame()

        # Numeric features only
        excluded = ['time', 'direction_label', 'upside_pct', 'downside_pct', 'future_drawdown_pct', 'label', 'index']
        cols = [c for c in self.df.select_dtypes(include=[np.number]).columns if c not in excluded]
        
        relevance_data = []
        target_vals = self.df[target_col].values
        
        for col in cols:
            # Drop NaNs for this specific correlation check
            valid_mask = ~self.df[col].isna()
            if valid_mask.sum() < 10: continue
            
            corr, _ = spearmanr(self.df.loc[valid_mask, col], target_vals[valid_mask])
            relevance_data.append({
                'Feature': col,
                'Relevance_Score': abs(corr),
                'Direction': 'Positive' if corr > 0 else 'Negative'
            })
            
        res_df = pd.DataFrame(relevance_data)
        return res_df.sort_values(by='Relevance_Score', ascending=False)

    def print_health_report(self):
        """Prints a comprehensive diagnostic report to the console"""
        relevance = self.get_feature_relevance()
        if relevance.empty:
            console.print("[error]❌ Could not calculate relevance. Check target columns.[/error]")
            return

        # Top 10 High Signal
        top_table = Table(title="🔥 TOP 10 MOST POWERFUL INDICATORS (High Signal)", show_header=True, header_style="bold green")
        top_table.add_column("Feature", style="cyan")
        top_table.add_column("Score", justify="right")
        top_table.add_column("Direction", justify="center")
        
        for _, row in relevance.head(10).iterrows():
            top_table.add_row(row['Feature'], f"{row['Relevance_Score']:.4f}", row['Direction'])

        # Bottom 10 Noise
        bottom_table = Table(title="💤 TOP 10 NOISE INDICATORS (Low/No Signal)", show_header=True, header_style="bold yellow")
        bottom_table.add_column("Feature", style="bright_black")
        bottom_table.add_column("Score", justify="right")
        bottom_table.add_column("Direction", justify="center")
        
        for _, row in relevance.tail(10).iterrows():
            bottom_table.add_row(row['Feature'], f"{row['Relevance_Score']:.4f}", row['Direction'])

        console.print(Panel(f"✅ Total Features Scanned: [bold]{len(relevance)}[/bold]", border_style="success"))
        console.print(top_table)
        console.print(bottom_table)
        
        # Check for constant features (zero variance)
        constant_features = [c for c in self.df.select_dtypes(include=[np.number]).columns if self.df[c].std() == 0]
        if constant_features:
            console.print(f"\n[error]⚠️ WARNING: Found {len(constant_features)} constant features (Zero Variance).[/error]")
            console.print(f"Examples: {constant_features[:5]}")
            
        console.print("\n[info]💡 Insight: Focus on features with 'Relevance_Score' > 0.05 for best AI results.[/info]")

    def prune_weak_features(self, threshold: float = 0.01) -> List[str]:
        """Returns list of features that are likely just noise"""
        relevance = self.get_feature_relevance()
        weak = relevance[relevance['Relevance_Score'] < threshold]['Feature'].tolist()
        return weak
