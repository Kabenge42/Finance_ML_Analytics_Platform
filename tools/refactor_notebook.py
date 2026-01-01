import sys
from pathlib import Path

# New Code Blocks
CODE_7_5 = r'''#%%
# ============================================================================
# Cell 7.5: Enhanced Financial Metrics & Price Target Analytics
# ============================================================================
from dataclasses import dataclass, field
from typing import Optional, List, Dict
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.express as px

from finance_ml.core.schema import PHASE93_FEATURE_CATEGORIES
from finance_ml.etl.stages.metrics import (
    compute_valuation_metrics,
    compute_profitability_metrics,
    compute_growth_metrics,
    compute_leverage_metrics,
    compute_target_vs_price_metrics,
    generate_metrics_dashboard
)

@dataclass
class FinancialMetricsConfig:
    """Configuration for financial metrics analytics."""
    min_price_target_count: int = 5
    compute_missing_metrics: bool = True
    valuation_metrics: List[str] = field(default_factory=lambda: [
        'p_e', 'p_s', 'ev_ebitda', 'ev_sales', 'eps'
    ])
    profitability_metrics: List[str] = field(default_factory=lambda: [
        'gross_margin', 'operating_margin', 'net_margin', 'roe', 'roa'
    ])
    verbose: bool = True

class FinancialMetricsAnalyzer:
    """Encapsulates financial metrics computation and price target analysis."""

    def __init__(self, df: pd.DataFrame, output_dir: Path, config: Optional[FinancialMetricsConfig] = None):
        self.df = df
        self.output_dir = Path(output_dir)
        self.config = config or FinancialMetricsConfig()
        self.metrics_summary: Dict[str, int] = {}
        self.dashboard_data: Dict = {}

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(message)

    def compute_metrics(self) -> 'FinancialMetricsAnalyzer':
        """Compute all standard financial metrics if configured."""
        if not self.config.compute_missing_metrics:
            return self

        self._log('\n🔧 Computing Financial Metrics...')
        
        # Valuation
        self.df = compute_valuation_metrics(self.df)
        self.metrics_summary['valuation'] = self._count_present(self.config.valuation_metrics)
        
        # Profitability
        self.df = compute_profitability_metrics(self.df)
        self.metrics_summary['profitability'] = self._count_present(self.config.profitability_metrics)
        
        # Growth
        self.df = compute_growth_metrics(self.df)
        
        # Leverage
        self.df = compute_leverage_metrics(self.df)
        
        # Target vs Price
        self.df = compute_target_vs_price_metrics(self.df)
        
        self._log(f"  ✓ Metrics computed. Summary: {self.metrics_summary}")
        return self

    def _count_present(self, cols: List[str]) -> int:
        return sum(1 for c in cols if c in self.df.columns)

    def analyze_price_targets(self) -> 'FinancialMetricsAnalyzer':
        """Analyze price targets and upside potential."""
        self._log('\n🎯 Analyzing Price Targets...')
        
        target_col = 'target_vs_price'
        if target_col not in self.df.columns:
            self._log("  ⚠️ Target vs Price column missing.")
            return self

        valid_targets = self.df[target_col].dropna()
        if len(valid_targets) == 0:
            return self

        stats = {
            'count': len(valid_targets),
            'mean_upside': float(valid_targets.mean()),
            'median_upside': float(valid_targets.median()),
            'positive_upside_pct': float((valid_targets > 0).mean() * 100)
        }
        
        self._log(f"  Stocks with Targets: {stats['count']:,}")
        self._log(f"  Mean Upside:         {stats['mean_upside']:+.2f}%")
        self._log(f"  Positive Upside:     {stats['positive_upside_pct']:.1f}%")
        
        self.dashboard_data['price_target_stats'] = stats
        return self

    def generate_dashboard(self) -> 'FinancialMetricsAnalyzer':
        """Generate and save the financial metrics dashboard."""
        self._log('\n💾 Generating Financial Metrics Dashboard...')
        
        dashboard = generate_metrics_dashboard(
            self.df,
            sector_column='sector',
            region_column='region' if 'region' in self.df.columns else None
        )
        
        # Merge local analytics
        dashboard.update(self.dashboard_data)
        
        output_path = self.output_dir / 'financial_metrics_dashboard.json'
        # Note: In real usage, ensure make_serializable is imported or handled
        import json
        with open(output_path, 'w') as f:
            json.dump(dashboard, f, indent=2, default=str)
            
        self._log(f'  ✓ Saved: {output_path}')
        return self

    def run_full_analysis(self) -> pd.DataFrame:
        """Run the complete pipeline."""
        self._log('=' * 80)
        self._log('ENHANCED FINANCIAL METRICS & PRICE TARGET ANALYTICS')
        self._log('=' * 80)
        
        (self.compute_metrics()
             .analyze_price_targets()
             .generate_dashboard())
             
        return self.df

# Execute
analyzer = FinancialMetricsAnalyzer(all_stocks_features, financial_metrics_dir)
all_stocks_features = analyzer.run_full_analysis()
'''

CODE_7_6 = r'''#%%
# ============================================================================
# Cell 7.6: Earnings Monitoring (Phase 9.3 Schema-Driven)
# ============================================================================
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import json

@dataclass
class EarningsMonitorConfig:
    """Configuration for earnings monitoring."""
    categories_to_monitor: Dict[str, str] = field(default_factory=lambda: {
        'Growth Metrics': 'growth',
        'Profitability': 'profitability',
        'Valuation Ratios': 'valuation',
        'Earnings Quality': 'earnings_quality',
        'Revenue Forecasting': 'forecasts',
        'Momentum & Technical': 'momentum'
    })
    top_n_metrics: int = 5
    verbose: bool = True

class EarningsMonitorAnalyzer:
    """Analyzes earnings-related metrics based on Phase 9.3 Schema."""

    def __init__(self, df: pd.DataFrame, output_dir: Path, config: Optional[EarningsMonitorConfig] = None):
        self.df = df
        self.output_dir = Path(output_dir)
        self.config = config or EarningsMonitorConfig()
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'total_stocks': len(df),
            'metrics': {}
        }

    def _log(self, message: str) -> None:
        if self.config.verbose:
            print(message)

    def _compute_metric_stats(self, series: pd.Series) -> Optional[Dict]:
        """Compute standardized statistics for a metric."""
        clean_data = pd.to_numeric(series, errors='coerce').dropna()
        if len(clean_data) == 0:
            return None
        
        return {
            'count': int(len(clean_data)),
            'mean': float(clean_data.mean()),
            'median': float(clean_data.median()),
            'std': float(clean_data.std()),
            'positive_pct': float((clean_data > 0).mean() * 100),
            'min': float(clean_data.min()),
            'max': float(clean_data.max())
        }

    def analyze_categories(self) -> 'EarningsMonitorAnalyzer':
        """Analyze all configured Phase 9.3 categories."""
        self._log('\n📈 Revenue & Earnings Growth Analysis (PHASE93_FEATURE_CATEGORIES)...')

        for schema_cat, monitor_key in self.config.categories_to_monitor.items():
            # Retrieve columns from Schema
            schema_cols = PHASE93_FEATURE_CATEGORIES.get(schema_cat, [])
            available_cols = [c for c in schema_cols if c in self.df.columns]
            
            if not available_cols:
                continue

            self._log(f'\n  📊 {schema_cat} ({len(available_cols)} metrics):')
            category_stats = {}

            # Analyze top N metrics
            for metric in available_cols[:self.config.top_n_metrics]:
                stats = self._compute_metric_stats(self.df[metric])
                if stats:
                    category_stats[metric] = stats
                    self._log(f"    {metric[:30]:30s}: mean={stats['mean']:>8.2f} | median={stats['median']:>8.2f}")
            
            self.results['metrics'][monitor_key] = category_stats
            
        return self

    def export_results(self) -> 'EarningsMonitorAnalyzer':
        """Export earnings monitor report."""
        output_path = self.output_dir / 'earnings_monitor.json'
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        self._log(f'\n💾 Saved: {output_path}')
        return self

    def run(self):
        """Execute full earnings monitoring workflow."""
        self._log('=' * 80)
        self._log('EARNINGS MONITORING (Phase 9.3 Schema-Driven)')
        self._log('=' * 80)
        
        self.analyze_categories().export_results()

# Execute
earnings_monitor = EarningsMonitorAnalyzer(all_stocks_features, financial_metrics_dir).run()
'''


def find_range(lines, start_marker, end_marker):
    start_idx = -1
    end_idx = -1
    for i, line in enumerate(lines):
        if start_marker in line:
            start_idx = i
            # Look backwards for #%%
            if i > 0 and lines[i - 1].strip() == "#%%":
                start_idx = i - 1
            if i > 1 and lines[i - 2].strip() == "#%%":  # handle decoration lines
                start_idx = i - 2
            break

    if start_idx == -1:
        return None

    for i in range(start_idx + 1, len(lines)):
        if end_marker in line:
            end_idx = i
            break

    if end_idx == -1:
        return None

    # Refine end: search backwards from end_idx for the last #%%
    # end_idx is the line with "## Cell 7.X" (Markdown).
    # We want to replace up to the line BEFORE "## Cell 7.X" start marker (#%% md)

    curr = end_idx
    while curr > start_idx:
        if lines[curr].strip().startswith("#%%"):
            end_idx = curr
            break
        curr -= 1

    return start_idx, end_idx


def process_notebook():
    nb_path = Path("etl_data_explorer.ipynb")
    lines = nb_path.read_text(encoding="utf-8").splitlines()

    # Locate Cell 7.5
    range_7_5 = find_range(
        lines, "# Cell 7.5: Enhanced Financial Metrics", "## Cell 7.6: Earnings Monitoring"
    )
    if not range_7_5:
        print("Could not find range for Cell 7.5")
        # Debug
        print("Searching for start: '# Cell 7.5: Enhanced Financial Metrics'")
        found_start = False
        for line in lines:
            if "# Cell 7.5: Enhanced Financial Metrics" in line:
                print(f"Found start line: {line}")
                found_start = True
                break
        if not found_start:
            print("Start marker NOT found")

        print("Searching for end: '## Cell 7.6: Earnings Monitoring'")
        found_end = False
        for line in lines:
            if "## Cell 7.6: Earnings Monitoring" in line:
                print(f"Found end line: {line}")
                found_end = True
                break
        if not found_end:
            print("End marker NOT found")

        return

    print(f"Cell 7.5 range: {range_7_5}")

    # Locate Cell 7.6
    range_7_6 = find_range(lines, "# Cell 7.6: Earnings Monitoring", "## Cell 7.7: Analyst Rating")
    if not range_7_6:
        print("Could not find range for Cell 7.6")
        return

    print(f"Cell 7.6 range: {range_7_6}")

    s1, e1 = range_7_5
    s2, e2 = range_7_6

    if s2 <= e1:
        print("Overlapping ranges!")
        return

    final_lines = (
        lines[:s1] + CODE_7_5.splitlines() + lines[e1:s2] + CODE_7_6.splitlines() + lines[e2:]
    )

    nb_path.write_text("\n".join(final_lines), encoding="utf-8")
    print("Notebook updated successfully.")


if __name__ == "__main__":
    process_notebook()
