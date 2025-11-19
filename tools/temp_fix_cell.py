print(f"\n✓ Selected {len(portfolio_candidates)} portfolio candidates")
if len(portfolio_candidates) > 0:
    print(f'  Sectors: {portfolio_candidates["sector"].nunique()}')
    print(f'  Average composite score: {portfolio_candidates["composite_score"].mean():.3f}')
    print("\nTop 10 Candidates:")
    display_cols = [
        "ticker",
        "sector",
        "market_cap",
        "composite_score",
        "expected_return",
        "mispricing_score",
    ]
    available_cols = [c for c in display_cols if c in portfolio_candidates.columns]
    print(portfolio_candidates[available_cols].head(100).to_string(index=False))
else:
    print("\n⚠️  all_stocks_phase95 not available, using top_candidates from Section 10")
    if "top_candidates" in dir():
        portfolio_candidates = top_candidates.head(100)
        print(f"✓ Using {len(portfolio_candidates)} candidates from top_candidates")
    else:
        print("⚠️  Skipping stock selection - no suitable dataframe available")
        portfolio_candidates = None
