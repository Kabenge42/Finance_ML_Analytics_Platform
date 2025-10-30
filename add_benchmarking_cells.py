#!/usr/bin/env python3
"""Add Phase 9.2 Benchmarking demonstration cells to ml_finance_model_main.ipynb."""

import json
from pathlib import Path


def add_benchmarking_cells():
    """Add benchmarking demonstration cells to the notebook."""

    notebook_path = Path("ml_finance_model_main.ipynb")

    # Read notebook
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    # Find insertion point - after last Phase 9.2 cell
    insert_idx = None
    for i, cell in enumerate(nb["cells"]):
        source = "".join(cell.get("source", []))
        if "Phase 9.2" in source or "phase92" in source.lower():
            insert_idx = i + 1

    if insert_idx is None:
        print("Could not find Phase 9.2 section. Appending to end.")
        insert_idx = len(nb["cells"])

    # New cells to add
    new_cells = [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "#### Phase 9.2 Benchmarking: Sector and Regional Analysis\n",
                "\n",
                "Demonstrating newly implemented benchmarking functions:\n",
                "1. **Sector Distribution Comparisons** - Compare valuation metrics across sectors\n",
                "2. **Regional Valuation Comparisons** - Statistical tests for regional differences\n",
                "3. **Peer Group Analysis** - Find and compare to similar companies\n",
                "4. **Time-Series Trend Analysis** - Detect metric trends over time",
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Example 1: Sector Distribution Comparisons\n",
                "from finance_ml import compare_sector_distributions\n",
                "\n",
                'print("\\n📊 Sector Distribution Comparisons\\n" + "="*50)\n',
                "\n",
                "# Compare P/E and P/B ratios across sectors\n",
                "metrics_to_compare = ['p_e', 'p_b', 'ev_ebitda', 'operating_margin']\n",
                "available_metrics = [m for m in metrics_to_compare if m in all_stocks.columns]\n",
                "\n",
                "if len(available_metrics) >= 2 and 'sector' in all_stocks.columns:\n",
                "    sector_dist = compare_sector_distributions(\n",
                "        all_stocks, \n",
                "        metrics=available_metrics[:2],  # Use first 2 available metrics\n",
                "        sector_column='sector'\n",
                "    )\n",
                "    \n",
                "    if not sector_dist.empty:\n",
                "        print(f\"\\nAnalyzed {len(sector_dist['sector'].unique())} sectors\")\n",
                '        print(f"\\nSample results for {available_metrics[0].upper()}:")\n',
                "        \n",
                "        # Display first metric results\n",
                "        metric_df = sector_dist[sector_dist['metric'] == available_metrics[0]]\n",
                "        metric_df_sorted = metric_df.sort_values('median')\n",
                "        \n",
                '        print("\\nSector Rankings by Median:")\n',
                "        for _, row in metric_df_sorted.head(5).iterrows():\n",
                "            print(f\"  {row['sector']:20s} | Median: {row['median']:7.2f} | Mean: {row['mean']:7.2f} | Count: {row['count']:3.0f}\")\n",
                "        \n",
                "        # Identify attractive sectors (low valuation)\n",
                "        attractive = metric_df_sorted.head(3)['sector'].tolist()\n",
                "        print(f\"\\n💡 Most attractive sectors (lowest {available_metrics[0].upper()}): {', '.join(attractive)}\")\n",
                "    else:\n",
                '        print("⚠ No sector distribution data available")\n',
                "else:\n",
                '    print("⚠ Need at least 2 metrics and sector column for comparison")',
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Example 2: Regional Valuation Comparisons with Statistical Tests\n",
                "from finance_ml import compare_regional_valuations\n",
                "\n",
                'print("\\n🌍 Regional Valuation Comparisons\\n" + "="*50)\n',
                "\n",
                "if 'region' in all_stocks.columns and len(available_metrics) > 0:\n",
                "    # Compare regions with statistical significance tests\n",
                "    regional_result = compare_regional_valuations(\n",
                "        all_stocks,\n",
                "        metrics=[available_metrics[0]],  # Use first available metric\n",
                "        region_column='region',\n",
                "        include_tests=True,\n",
                "        test_method='anova'\n",
                "    )\n",
                "    \n",
                "    if isinstance(regional_result, dict) and 'distributions' in regional_result:\n",
                "        distributions = regional_result['distributions']\n",
                "        \n",
                "        if not distributions.empty:\n",
                '            print(f"\\nRegional averages for {available_metrics[0].upper()}:")\n',
                "            for _, row in distributions.iterrows():\n",
                "                print(f\"  {row['region']:10s} | Mean: {row['mean']:7.2f} | Median: {row['median']:7.2f} | Count: {row['count']:4.0f}\")\n",
                "            \n",
                "            # Display statistical test results\n",
                "            if 'statistical_tests' in regional_result:\n",
                "                tests = regional_result['statistical_tests']\n",
                "                for metric, test_result in tests.items():\n",
                '                    print(f"\\n📈 Statistical Test for {metric.upper()}:")\n',
                "                    print(f\"  Method: {test_result['method']}\")\n",
                "                    print(f\"  Test Statistic: {test_result['statistic']:.4f}\")\n",
                "                    print(f\"  P-value: {test_result['p_value']:.4f}\")\n",
                "                    \n",
                "                    if test_result['significant']:\n",
                '                        print(f"  ✓ Result: Significant regional differences detected (p < 0.05)")\n',
                "                    else:\n",
                '                        print(f"  → Result: No significant regional differences (p ≥ 0.05)")\n',
                "        else:\n",
                '            print("⚠ No regional comparison data available")\n',
                "    else:\n",
                '        print("⚠ Regional comparison failed")\n',
                "else:\n",
                '    print("⚠ Need region column and metrics for regional comparison")',
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Example 3: Peer Group Analysis\n",
                "from finance_ml import find_peer_group, compare_to_peers\n",
                "\n",
                'print("\\n👥 Peer Group Analysis\\n" + "="*50)\n',
                "\n",
                "if 'ticker' in all_stocks.columns and 'sector' in all_stocks.columns:\n",
                "    # Select a target stock (use first one as example)\n",
                "    sample_ticker = all_stocks['ticker'].iloc[0] if len(all_stocks) > 0 else None\n",
                "    \n",
                "    if sample_ticker:\n",
                '        print(f"\\nAnalyzing peer group for: {sample_ticker}")\n',
                "        \n",
                "        # Find peer group by market cap similarity\n",
                "        peers = find_peer_group(\n",
                "            all_stocks,\n",
                "            ticker=sample_ticker,\n",
                "            n_peers=5,\n",
                "            criteria='market_cap' if 'market_cap' in all_stocks.columns else 'last_price',\n",
                "            sector_column='sector'\n",
                "        )\n",
                "        \n",
                "        if not peers.empty:\n",
                '            print(f"\\nFound {len(peers)} peers in same sector:")\n',
                "            for ticker in peers['ticker'].head(5):\n",
                '                print(f"  • {ticker}")\n',
                "            \n",
                "            # Compare target to peers on key metrics\n",
                "            comparison_metrics = [m for m in ['p_e', 'p_b'] if m in all_stocks.columns]\n",
                "            \n",
                "            if comparison_metrics:\n",
                "                comparison = compare_to_peers(\n",
                "                    all_stocks,\n",
                "                    ticker=sample_ticker,\n",
                "                    metrics=comparison_metrics,\n",
                "                    n_peers=5\n",
                "                )\n",
                "                \n",
                "                if comparison:\n",
                '                    print(f"\\n📊 Comparison to Peers:")\n',
                "                    for metric, stats in comparison.items():\n",
                "                        deviation_pct = stats['deviation_pct']\n",
                "                        z_score = stats['z_score']\n",
                "                        \n",
                '                        print(f"\\n  {metric.upper()}:")\n',
                "                        print(f\"    {sample_ticker}: {stats['target']:.2f}\")\n",
                "                        print(f\"    Peers avg: {stats['peers_mean']:.2f}\")\n",
                '                        print(f"    Deviation: {deviation_pct:+.1f}% (z-score: {z_score:+.2f})")\n',
                "                        \n",
                "                        if abs(deviation_pct) > 20:\n",
                '                            direction = "undervalued" if deviation_pct < 0 else "overvalued"\n',
                '                            print(f"    💡 {sample_ticker} appears {direction} on {metric.upper()} (>20% deviation)")\n',
                "                else:\n",
                '                    print("⚠ No comparison data available")\n',
                "        else:\n",
                '            print(f"⚠ No peers found for {sample_ticker}")\n',
                "    else:\n",
                '        print("⚠ No tickers available for analysis")\n',
                "else:\n",
                '    print("⚠ Need ticker and sector columns for peer analysis")',
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Example 4: Time-Series Trend Analysis (if temporal data available)\n",
                "from finance_ml import analyze_metric_trend\n",
                "\n",
                'print("\\n📈 Time-Series Trend Analysis\\n" + "="*50)\n',
                "\n",
                "# Check if we have a date column\n",
                "date_columns = [col for col in all_stocks.columns if 'date' in col.lower()]\n",
                "\n",
                "if date_columns and 'ticker' in all_stocks.columns:\n",
                "    date_col = date_columns[0]\n",
                "    sample_ticker = all_stocks['ticker'].iloc[0] if len(all_stocks) > 0 else None\n",
                "    \n",
                "    if sample_ticker and len(available_metrics) > 0:\n",
                "        # Try to analyze trend for first available metric\n",
                "        trend = analyze_metric_trend(\n",
                "            all_stocks,\n",
                "            ticker=sample_ticker,\n",
                "            metric=available_metrics[0],\n",
                "            date_column=date_col\n",
                "        )\n",
                "        \n",
                "        if trend:\n",
                '            print(f"\\nTrend analysis for {sample_ticker} - {available_metrics[0].upper()}:")\n',
                "            print(f\"  Direction: {trend['trend_direction'].upper()}\")\n",
                "            print(f\"  Slope: {trend['slope']:.4f}\")\n",
                "            print(f\"  R²: {trend['r_squared']:.3f}\")\n",
                "            print(f\"  P-value: {trend['p_value']:.4f}\")\n",
                "            print(f\"  Periods: {trend['n_periods']}\")\n",
                "            \n",
                "            # Interpret results\n",
                "            if trend['trend_direction'] == 'increasing' and trend['r_squared'] > 0.7:\n",
                '                print(f"\\n  💡 Strong upward trend detected - valuation may be overheating")\n',
                "            elif trend['trend_direction'] == 'decreasing' and trend['r_squared'] > 0.7:\n",
                '                print(f"\\n  💡 Strong downward trend detected - potential opportunity")\n',
                "            elif trend['trend_direction'] == 'stable':\n",
                '                print(f"\\n  → Stable trend - no significant change over time")\n',
                "        else:\n",
                '            print("⚠ Insufficient data for trend analysis (need at least 3 time points)")\n',
                "    else:\n",
                '        print("⚠ Need ticker and metrics for trend analysis")\n',
                "else:\n",
                '    print("ℹ Time-series trend analysis requires a date column in the dataset")\n',
                '    print("  This feature will work when temporal data is available")\n',
                '    print("  Example: Multiple snapshots of stock data over time")',
            ],
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "source": [
                "# Example 5: Comprehensive Benchmarking Report\n",
                "from finance_ml import generate_benchmarking_report\n",
                "\n",
                'print("\\n📋 Comprehensive Benchmarking Report\\n" + "="*50)\n',
                "\n",
                "if len(available_metrics) > 0:\n",
                "    # Generate complete benchmarking report\n",
                "    report = generate_benchmarking_report(\n",
                "        all_stocks,\n",
                "        metrics=available_metrics[:3],  # Use first 3 available metrics\n",
                "        sector_column='sector',\n",
                "        region_column='region'\n",
                "    )\n",
                "    \n",
                "    # Display summary\n",
                '    print("\\n📊 Report Summary:")\n',
                "    print(f\"  Total stocks analyzed: {report['summary']['total_stocks']}\")\n",
                "    print(f\"  Number of sectors: {report['summary']['n_sectors']}\")\n",
                "    print(f\"  Number of regions: {report['summary']['n_regions']}\")\n",
                "    print(f\"  Metrics analyzed: {', '.join(report['summary']['metrics_analyzed'])}\")\n",
                "    \n",
                "    # Display sector distribution insights\n",
                "    if report['sector_distributions']:\n",
                "        print(f\"\\n  ✓ Sector distributions: {len(report['sector_distributions'])} entries\")\n",
                '        print("    (Detailed statistics for each sector-metric combination)")\n',
                "    \n",
                "    # Display regional valuation insights\n",
                "    if report['regional_valuations']:\n",
                "        print(f\"  ✓ Regional valuations: {len(report['regional_valuations'])} entries\")\n",
                '        print("    (Comparative statistics across regions)")\n',
                "    \n",
                '    print("\\n💡 Key Features:")\n',
                '    print("   • Sector-wise distribution analysis for valuation metrics")\n',
                '    print("   • Regional performance comparisons with statistical tests")\n',
                '    print("   • Peer group identification and relative valuation analysis")\n',
                '    print("   • Time-series trend detection for metric evolution")\n',
                "    \n",
                '    print("\\n✅ Phase 9.2 Benchmarking demonstrations complete!")\n',
                "else:\n",
                '    print("⚠ Need valuation metrics for benchmarking report")',
            ],
        },
    ]

    # Insert cells
    for i, cell in enumerate(new_cells):
        nb["cells"].insert(insert_idx + i, cell)

    # Write back
    with open(notebook_path, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"\n✓ Added {len(new_cells)} benchmarking demonstration cells to notebook")
    print(f"  Inserted at position {insert_idx}")
    print(f"  Total cells now: {len(nb['cells'])}")
    print("\nCells demonstrate:")
    print("  1. Sector distribution comparisons")
    print("  2. Regional valuation comparisons with statistical tests")
    print("  3. Peer group analysis and relative valuation")
    print("  4. Time-series trend analysis")
    print("  5. Comprehensive benchmarking report")

    return True


if __name__ == "__main__":
    add_benchmarking_cells()
