#!/usr/bin/env python3
"""Apply comprehensive refactorings to Phase 9.7 of ml_finance_model_main.ipynb."""

import json
import sys

# The refactored Phase 9.7 code with all improvements from the issue description
REFACTORED_PHASE97_CODE = '''# NOTE: All Phase 9.7 valuation functions are now imported in the main imports section above
# The following functions are available from finance_ml.eval:
#   - assign_valuation_category, calculate_sector_zscores, calculate_percentile_ranks
#   - calculate_multi_factor_score, rank_undervalued_stocks, rank_overvalued_stocks
#   - filter_stocks_by_criteria, create_valuation_scatter_plot, create_sector_heatmap
#   - create_region_sector_heatmap, export_predictions_to_excel
# No additional imports needed here.

# Constants
DEFAULT_RISK_FREE_RATE = 0.04
DEFAULT_VOLATILITY = 0.20
TOP_N_RANKINGS = 10
MIN_LARGE_CAP_MARKET_CAP = 10.0
MIN_STRONG_UPSIDE_PERCENT = 10.0
MIN_TECH_UPSIDE_PERCENT = 15.0
VALUATION_CATEGORIES_BUY = ['Strong Buy', 'Buy']

# Factor weights for multi-factor scoring
FACTOR_WEIGHTS = {
    'valuation': 0.5,
    'quality': 0.3,
    'growth': 0.2
}

# Required columns for analysis
REQUIRED_COLUMNS = {
    'prerequisites': ['predicted_price_target'],
    'sector_analysis': ['sector'],
    'display_leaders': ['Sector', 'Ticker', 'Company Name', 'Last Price', 'mispricing_score'],
    'screening': ['market_cap']
}


def verify_prerequisites():
    """Verify that all required data and configuration are available."""
    if 'config' not in globals():
        print("\\n⚠ Error: 'config' not found. Please run earlier phases to initialize configuration.")
        return False
    
    if 'all_stocks_featured' not in globals():
        print("\\n⚠ Error: 'all_stocks_featured' not found. Please run Phase 9.6 first.")
        return False
    
    if not hasattr(config, 'output_dir'):
        print("\\n⚠ Error: 'config.output_dir' not configured.")
        return False
    
    if 'predicted_price_target' not in all_stocks_featured.columns:
        print("\\n⚠ No predictions available. Run Phase 9.5 first.")
        return False
    
    if len(all_stocks_featured) == 0:
        print("\\n⚠ Warning: Dataset is empty. No stocks to analyze.")
        return False
    
    return True


def calculate_valuation_metrics(stocks_df):
    """Calculate core valuation metrics including mispricing scores and risk adjustments."""
    print("\\n💰 Calculating Mispricing Scores...")
    valued_stocks = stocks_df.copy()
    valued_stocks['mispricing_score'] = calculate_mispricing_score(stocks_df)
    
    print("\\n📊 Calculating Risk-Adjusted Mispricing...")
    risk_adjusted = calculate_risk_adjusted_mispricing(
        valued_stocks,
        risk_free_rate=DEFAULT_RISK_FREE_RATE,
        use_confidence_interval=False,
        default_volatility=DEFAULT_VOLATILITY
    )
    valued_stocks['risk_adjusted_mispricing'] = risk_adjusted
    
    print("\\n📊 Assigning Valuation Categories...")
    categories = assign_valuation_category(valued_stocks['mispricing_score'])
    valued_stocks['valuation_category'] = categories
    
    display_valuation_distribution(valued_stocks)
    return valued_stocks


def display_valuation_distribution(stocks_df):
    """Display distribution of valuation categories."""
    print("\\n📈 Valuation Category Distribution:")
    category_counts = stocks_df['valuation_category'].value_counts()
    total_stocks = len(stocks_df)
    
    for category, count in category_counts.items():
        pct = (count / total_stocks) * 100 if total_stocks > 0 else 0
        print(f"  {category}: {count:,} stocks ({pct:.1f}%)")


def perform_sector_analysis(stocks_df):
    """Perform sector-relative valuation analysis including z-scores and percentiles."""
    if 'sector' not in stocks_df.columns:
        print("\\n⚠ 'sector' column not found - skipping sector-relative analysis")
        return stocks_df
    
    valuation_metrics = get_available_valuation_metrics(stocks_df)
    if not valuation_metrics:
        print("  ⚠ No valuation metrics (p_e, p_b, ev_ebitda) found for z-score calculation")
        return stocks_df
    
    stocks_df = calculate_and_apply_zscores(stocks_df, valuation_metrics)
    stocks_df = calculate_and_apply_percentiles(stocks_df, valuation_metrics)
    
    return stocks_df


def calculate_and_apply_zscores(stocks_df, valuation_metrics):
    """Calculate sector-relative z-scores and apply to dataframe."""
    print("\\n📈 Calculating Sector-Relative Valuation (Z-Scores)...")
    zscores_df = calculate_sector_zscores(stocks_df, valuation_metrics, sector_col='sector')
    
    for col in zscores_df.columns:
        stocks_df[col] = zscores_df[col]
    
    print(f"  ✓ Calculated z-scores for: {', '.join(valuation_metrics)}")
    return stocks_df


def calculate_and_apply_percentiles(stocks_df, valuation_metrics):
    """Calculate percentile ranks within sectors and apply to dataframe."""
    print("\\n📊 Calculating Percentile Ranks within Sectors...")
    percentiles_df = calculate_percentile_ranks(stocks_df, valuation_metrics, sector_col='sector')
    
    for col in percentiles_df.columns:
        stocks_df[col] = percentiles_df[col]
    
    print(f"  ✓ Calculated percentile ranks for: {', '.join(valuation_metrics)}")
    return stocks_df


def get_available_valuation_metrics(stocks_df):
    """Get list of available valuation metrics from the dataframe."""
    possible_metrics = ['p_e', 'p_b', 'ev_ebitda']
    return [metric for metric in possible_metrics if metric in stocks_df.columns]


def calculate_multi_factor_scores(stocks_df):
    """Calculate multi-factor scores combining valuation, quality, and growth."""
    print("\\n🎯 Calculating Multi-Factor Scores...")
    
    quality_cols = get_available_columns(stocks_df, ['roe', 'ebitda_margin'])
    growth_cols = get_available_columns(stocks_df, ['revenue_growth'])
    
    multi_factor_score = calculate_multi_factor_score(
        stocks_df,
        valuation_col='mispricing_score',
        quality_cols=quality_cols if quality_cols else None,
        growth_cols=growth_cols if growth_cols else None,
        weights=FACTOR_WEIGHTS
    )
    stocks_df['multi_factor_score'] = multi_factor_score
    
    print(f"  ✓ Combined valuation, quality, and growth into composite score")
    return stocks_df


def get_available_columns(stocks_df, column_names):
    """Get list of columns that exist in the dataframe."""
    return [col for col in column_names if col in stocks_df.columns]


def display_rankings(stocks_df):
    """Display top undervalued and overvalued stocks."""
    print(f"\\n🏆 Top {TOP_N_RANKINGS} Undervalued Stocks (Buy Opportunities):")
    top_undervalued = rank_undervalued_stocks(stocks_df, top_n=TOP_N_RANKINGS)
    display_stock_ranking(top_undervalued)
    
    print(f"\\n⚠️  Top {TOP_N_RANKINGS} Overvalued Stocks (Sell/Short Opportunities):")
    top_overvalued = rank_overvalued_stocks(stocks_df, top_n=TOP_N_RANKINGS)
    display_stock_ranking(top_overvalued)


def display_stock_ranking(ranked_stocks):
    """Display formatted stock ranking information."""
    for i, row in ranked_stocks.iterrows():
        ticker = row.get('ticker', 'N/A')
        sector = row.get('sector', 'N/A')
        mispricing = row.get('mispricing_score', 0)
        category = row.get('valuation_category', 'N/A')
        print(f"  {ticker:<10s} | {sector:<25s} | {mispricing:>6.1f}% | {category}")


def display_sector_leaders_laggards(all_stocks_valued):
    """Display sector leaders and laggards with proper error handling."""
    import pandas as pd
    
    missing_cols = validate_required_columns(all_stocks_valued, REQUIRED_COLUMNS['display_leaders'])
    if missing_cols:
        print(f"Warning: Missing required columns: {missing_cols}")
        return
    
    sector_analysis = analyze_sectors(all_stocks_valued)
    display_sector_analysis_results(sector_analysis)
    
    return sector_analysis


def validate_required_columns(df, required_columns):
    """Validate that dataframe contains required columns."""
    return [col for col in required_columns if col not in df.columns]


def analyze_sectors(all_stocks_valued):
    """Analyze each sector to identify leaders and laggards."""
    import pandas as pd
    sector_analysis = {}
    
    for sector in all_stocks_valued['Sector'].unique():
        if pd.isna(sector):
            continue
        
        sector_data = all_stocks_valued[all_stocks_valued['Sector'] == sector].copy()
        if len(sector_data) == 0:
            continue
        
        sector_analysis[sector] = create_sector_summary(sector_data)
    
    return sector_analysis


def create_sector_summary(sector_data):
    """Create summary statistics for a sector including leaders and laggards."""
    sector_data_sorted = sector_data.sort_values('mispricing_score', ascending=False)
    
    display_columns = ['Ticker', 'Company Name', 'Last Price', 'mispricing_score']
    leaders = sector_data_sorted.head(5)[display_columns]
    laggards = sector_data_sorted.tail(5)[display_columns]
    
    return {
        'leaders': leaders,
        'laggards': laggards,
        'count': len(sector_data),
        'avg_mispricing': sector_data['mispricing_score'].mean()
    }


def display_sector_analysis_results(sector_analysis):
    """Display formatted sector analysis results."""
    print("\\n" + "="*80)
    print("SECTOR LEADERS & LAGGARDS ANALYSIS")
    print("="*80)
    
    for sector in sorted(sector_analysis.keys()):
        analysis = sector_analysis[sector]
        
        if 'leaders' not in analysis or 'laggards' not in analysis:
            print(f"\\nWarning: Incomplete data for sector '{sector}', skipping...")
            continue
        
        display_single_sector_analysis(sector, analysis)
    
    print("="*80)
    print(f"Analysis complete. Processed {len(sector_analysis)} sectors.")
    print("="*80)


def display_single_sector_analysis(sector, analysis):
    """Display analysis for a single sector."""
    print(f"\\n{'─'*80}")
    print(f"SECTOR: {sector}")
    print(f"Total Stocks: {analysis.get('count', 0)} | "
          f"Avg Mispricing Score: {analysis.get('avg_mispricing', 0):.4f}")
    print(f"{'─'*80}")
    
    print(f"\\n🟢 TOP 5 UNDERVALUED (Leaders):")
    display_sector_group(analysis['leaders'], "leaders")
    
    print(f"\\n🔴 TOP 5 OVERVALUED (Laggards):")
    display_sector_group(analysis['laggards'], "laggards")
    print()


def display_sector_group(stocks_df, group_type):
    """Display a group of stocks (leaders or laggards)."""
    if len(stocks_df) > 0:
        print(stocks_df.to_string(index=False))
    else:
        print(f"  No {group_type} identified for this sector")


def display_stock_screening_examples(stocks_df):
    """Display examples of filtered stock screens."""
    print("\\n🔍 Stock Screening Examples:")
    
    display_large_cap_screen(stocks_df)
    display_tech_sector_screen(stocks_df)


def display_large_cap_screen(stocks_df):
    """Display large-cap undervalued stocks screen."""
    if 'market_cap' not in stocks_df.columns:
        print("  ⚠ 'market_cap' column not found - skipping large-cap analysis")
        return
    
    large_cap_undervalued = filter_stocks_by_criteria(
        stocks_df,
        min_market_cap=MIN_LARGE_CAP_MARKET_CAP,
        min_mispricing=MIN_STRONG_UPSIDE_PERCENT,
        valuation_categories=VALUATION_CATEGORIES_BUY
    )
    print(f"  • Large-cap undervalued (>${MIN_LARGE_CAP_MARKET_CAP}B, >"
          f"{MIN_STRONG_UPSIDE_PERCENT}% upside): {len(large_cap_undervalued)} stocks")


def display_tech_sector_screen(stocks_df):
    """Display technology sector opportunities screen."""
    if 'sector' not in stocks_df.columns:
        return
    
    tech_sector_names = find_tech_sectors(stocks_df)
    if not tech_sector_names:
        print("  ⚠ No technology sector found in data")
        return
    
    tech_opportunities = filter_stocks_by_criteria(
        stocks_df,
        sectors=tech_sector_names,
        min_mispricing=MIN_TECH_UPSIDE_PERCENT,
        valuation_categories=VALUATION_CATEGORIES_BUY
    )
    print(f"  • Technology sector strong opportunities: {len(tech_opportunities)} stocks")


def find_tech_sectors(stocks_df):
    """Find technology-related sectors in the dataframe."""
    unique_sectors = stocks_df['sector'].unique()
    return [s for s in unique_sectors 
            if 'tech' in str(s).lower() or 'information' in str(s).lower()]


def generate_reports_and_visualizations(stocks_df):
    """Generate all output files including visualizations and reports."""
    print("\\n📊 Creating Interactive Visualizations...")
    
    output_dir = setup_output_directory()
    if output_dir is None:
        return
    
    create_scatter_plot(stocks_df, output_dir)
    create_heatmaps(stocks_df, output_dir)
    export_excel_report(stocks_df, output_dir)
    generate_pdf_summary(stocks_df, output_dir)


def setup_output_directory():
    """Setup and validate output directory."""
    if not hasattr(config, 'output_dir'):
        print("  ⚠ Error: config.output_dir not configured. Cannot generate reports.")
        return None
    
    try:
        from pathlib import Path
        output_dir = config.analytics_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
    except (TypeError, AttributeError, OSError) as e:
        print(f"  ⚠ Error creating output directory: {str(e)}")
        return None


def create_scatter_plot(stocks_df, output_dir):
    """Create valuation scatter plot."""
    try:
        scatter_path = output_dir / 'valuation_scatter_plot.html'
        create_valuation_scatter_plot(stocks_df, out_path=scatter_path, color_by='sector')
        print(f"  ✓ Scatter plot (Price vs Target): {scatter_path}")
    except Exception as e:
        handle_visualization_error("scatter plot", e)


def create_heatmaps(stocks_df, output_dir):
    """Create sector and region-sector heatmaps."""
    create_sector_heatmap_viz(stocks_df, output_dir)
    create_region_sector_heatmap_viz(stocks_df, output_dir)


def create_sector_heatmap_viz(stocks_df, output_dir):
    """Create sector performance heatmap."""
    try:
        sector_heatmap_path = output_dir / 'sector_heatmap.png'
        create_sector_heatmap(stocks_df, out_path=sector_heatmap_path, metric='mispricing_score')
        print(f"  ✓ Sector heatmap: {sector_heatmap_path}")
    except Exception as e:
        handle_visualization_error("sector heatmap", e)


def create_region_sector_heatmap_viz(stocks_df, output_dir):
    """Create region-sector performance heatmap."""
    if 'region' not in stocks_df.columns or 'sector' not in stocks_df.columns:
        return
    
    try:
        region_sector_heatmap_path = output_dir / 'region_sector_heatmap.png'
        create_region_sector_heatmap(
            stocks_df,
            metric='mispricing_score',
            out_path=region_sector_heatmap_path
        )
        print(f"  ✓ Region×Sector heatmap: {region_sector_heatmap_path}")
    except Exception as e:
        handle_visualization_error("region-sector heatmap", e)


def export_excel_report(stocks_df, output_dir):
    """Export stock valuation analysis to Excel."""
    try:
        excel_path = output_dir / 'stock_valuation_analysis.xlsx'
        export_predictions_to_excel(stocks_df, excel_path, include_summary=True)
        print(f"  ✓ Excel report: {excel_path}")
    except Exception as e:
        handle_visualization_error("Excel report", e)


def generate_pdf_summary(stocks_df, output_dir):
    """Generate PDF summary report."""
    print("\\n📄 Generating PDF Report...")
    pdf_path = output_dir / 'stock_valuation_report.pdf'
    
    try:
        generate_pdf_report(
            stocks_df,
            pdf_path=pdf_path,
            title="Stock Valuation Analysis Report",
            include_summary=True,
            top_n_opportunities=20,
            include_charts=False
        )
        print(f"  ✓ PDF report: {pdf_path}")
    except ImportError:
        print("  ⚠ ReportLab not installed; skipping PDF report generation")
        print("    Install with: pip install reportlab")
    except Exception as e:
        handle_visualization_error("PDF report", e)


def handle_visualization_error(viz_type, error):
    """Handle visualization creation errors with consistent logging."""
    import logging
    print(f"  ⚠ Failed to create {viz_type}: {str(error)}")
    logging.exception(f"Detailed error in {viz_type}")


def print_completion_message():
    """Print completion status messages."""
    print("\\n✓ PHASE 9.7 COMPLETE — STOCK VALUATION AND IDENTIFICATION")
    print("✓ PHASE 9 COMPLETE — END-TO-END ML ANALYTICS PLATFORM")
    print("📊 Business Objective Achieved: Stock Price Target Predictions with Comprehensive Valuation Analysis")


# Main execution flow
print_section_header("PHASE 9.7 — VALUATION AND STOCK IDENTIFICATION")

if not verify_prerequisites():
    print("\\n⚠ Skipping Phase 9.7 - Prerequisites not met. Please run previous phases first.")
else:
    try:
        all_stocks_valued = calculate_valuation_metrics(all_stocks_featured)
        all_stocks_valued = perform_sector_analysis(all_stocks_valued)
        all_stocks_valued = calculate_multi_factor_scores(all_stocks_valued)
        
        display_rankings(all_stocks_valued)
        display_sector_leaders_laggards(all_stocks_valued)
        display_stock_screening_examples(all_stocks_valued)
        
        generate_reports_and_visualizations(all_stocks_valued)
        
        print_completion_message()
    except Exception as e:
        print(f"\\n❌ Error in Phase 9.7: {str(e)}")
        import traceback
        traceback.print_exc()
'''


def main():
    """Apply refactorings to the notebook."""
    notebook_path = "ml_finance_model_main.ipynb"

    print(f"Loading notebook: {notebook_path}")
    try:
        with open(notebook_path, "r", encoding="utf-8") as f:
            notebook = json.load(f)
    except Exception as e:
        print(f"Error loading notebook: {e}")
        return 1

    # Find the Phase 9.7 cell
    phase97_cell_index = None
    for idx, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if "PHASE 9.7 — VALUATION AND STOCK IDENTIFICATION" in source and len(source) > 10000:
                phase97_cell_index = idx
                print(f"Found Phase 9.7 cell at index {idx}")
                break

    if phase97_cell_index is None:
        print("Error: Phase 9.7 cell not found")
        return 1

    # Replace the cell content with refactored code
    print("Applying refactorings to Phase 9.7 cell...")
    # Split by newline and add newlines back (except for last line)
    lines = REFACTORED_PHASE97_CODE.split("\n")
    notebook["cells"][phase97_cell_index]["source"] = [line + "\n" for line in lines[:-1]] + (
        [lines[-1]] if lines[-1] else []
    )

    # Create backup
    backup_path = notebook_path + ".backup_refactoring"
    print(f"Creating backup: {backup_path}")
    with open(backup_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    # Save the updated notebook
    print(f"Saving refactored notebook: {notebook_path}")
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)

    print("\n✓ Successfully applied Phase 9.7 refactorings")
    print("\nRefactoring summary:")
    print("  • Extracted constants (DEFAULT_RISK_FREE_RATE, TOP_N_RANKINGS, etc.)")
    print("  • Split large functions into smaller, focused functions")
    print("  • Unified error handling patterns")
    print("  • Created reusable utility functions")
    print("  • Improved code readability and maintainability")

    return 0


if __name__ == "__main__":
    sys.exit(main())
