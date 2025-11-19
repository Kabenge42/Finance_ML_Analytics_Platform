# 10.1 Stock Selection – Advanced multi-criteria filtering and ML ranking
print("\n" + "=" * 80)
print("10.1 STOCK SELECTION - Advanced Filtering & Ranking")
print("=" * 80)

if "all_stocks_phase95" in dir() and not all_stocks_phase95.empty:
    print(f"\n✓ Using all_stocks_phase95 dataframe: {len(all_stocks_phase95)} stocks")

    # Pre-filter diagnostics to detect normalized vs absolute market cap
    print("\n📊 Pre-filter diagnostics:")
    print(f"  Total stocks: {len(all_stocks_phase95):,}")

    if "market_cap" in all_stocks_phase95.columns:
        mc = all_stocks_phase95["market_cap"].dropna()
        print(f"  Market cap available: {len(mc):,} stocks")
        if len(mc) > 0:
            # Check if data is normalized (range 0-1) or absolute
            is_normalized = (mc.min() >= 0) and (mc.max() <= 1.5)

            if is_normalized:
                print(f"    ✓ Market cap is NORMALIZED (0-1 scale)")
                print(f"    Range: {mc.min():.3f} to {mc.max():.3f}")
                print(f"    Median: {mc.median():.3f}")
                print(f"    75th percentile: {mc.quantile(0.75):.3f}")

                # Use normalized threshold for top 50% by market cap
                min_mc_threshold = 0.5
                cap_unit = ""  # No scaling needed for normalized data
            else:
                print(f"    ✓ Market cap is in ABSOLUTE units")
                print(f"    Range: ${mc.min() / 1e9:.2f}B to ${mc.max() / 1e9:.2f}B")
                print(f"    Median: ${mc.median() / 1e9:.2f}B")

                # Use absolute threshold
                min_mc_threshold = 1.0
                cap_unit = "B"
    else:
        print('  ⚠️  WARNING: "market_cap" column not found!')
        min_mc_threshold = None
        cap_unit = ""

    # Apply multi-criteria selection with auto-detected parameters
    print(f'\n🎯 Applying filters: min_market_cap={min_mc_threshold}, cap_unit="{cap_unit}"')

    portfolio_candidates = select_portfolio_candidates(
        all_stocks_phase95,
        min_market_cap=min_mc_threshold if min_mc_threshold is not None else 0.0,
        top_n=500,
        max_sector_weight=0.25,
        cap_unit=cap_unit,
    )

    if len(portfolio_candidates) == 0:
        print("\n⚠️  No candidates selected with current filters; relaxing market cap constraint...")
        # Retry with no market cap filter
        portfolio_candidates = select_portfolio_candidates(
            all_stocks_phase95, min_market_cap=0.0, top_n=500, max_sector_weight=0.25, cap_unit=""
        )

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
