"""
Visualization Code for First 50 Stocks from all_stocks_preprocessed
Generated: 2025-12-14
Data Source: ETL pipeline output (all_stocks_preprocessed.head(50))
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Style configuration (following code_guidelines.md Section 17)
plt.style.use("dark_background")

# Load the data (replace with your actual path to data)
# For demonstration, assuming data is already loaded as all_stocks_preprocessed
# all_stocks_preprocessed = pd.read_csv('path/to/your/data.csv')  # If loading from file
# df_50 = all_stocks_preprocessed.head(50)

# ============================================================================
# Visualization 1: Market Capitalization Distribution by Sector
# ============================================================================


def viz_market_cap_by_sector(df):
    """Bar chart showing average market capitalization by sector."""
    plt.figure(figsize=(12, 6))

    # Group by sector and calculate mean market cap
    sector_market_cap = (
        df.groupby("sector")["market_cap"].mean().sort_values(ascending=False)
    )

    # Create bar chart
    ax = sector_market_cap.plot(kind="bar", color="teal", edgecolor="black", alpha=0.8)

    # Formatting
    plt.title("Average Market Capitalization by Sector", fontsize=16, fontweight="bold")
    plt.ylabel("Market Capitalization (in Millions)", fontsize=12)
    plt.xlabel("Sector", fontsize=12)
    plt.xticks(rotation=45, ha="right", fontsize=10)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()

    # Add value labels on bars
    for i, v in enumerate(sector_market_cap.values):
        ax.text(i, v, f"${v:,.0f}M", ha="center", va="bottom", fontsize=9)

    return plt


# ============================================================================
# Visualization 2: Revenue vs EBITDA Scatter Plot
# ============================================================================


def viz_revenue_vs_ebitda(df):
    """Scatter plot showing relationship between revenue and EBITDA."""
    plt.figure(figsize=(12, 6))

    # Create scatter plot
    plt.scatter(
        df["total_revenues_ltm"],
        df["ebitda_ltm"],
        alpha=0.7,
        edgecolor="black",
        color="green",
        s=100,
    )

    # Formatting
    plt.title("Revenue vs EBITDA", fontsize=16, fontweight="bold")
    plt.xlabel("Revenue (in Millions)", fontsize=12)
    plt.ylabel("EBITDA (in Millions)", fontsize=12)
    plt.grid(linestyle="--", alpha=0.7)

    # Add trend line if data available
    valid_data = df[["total_revenues_ltm", "ebitda_ltm"]].dropna()
    if len(valid_data) > 1:
        z = np.polyfit(valid_data["total_revenues_ltm"], valid_data["ebitda_ltm"], 1)
        p = np.poly1d(z)
        plt.plot(
            valid_data["total_revenues_ltm"].sort_values(),
            p(valid_data["total_revenues_ltm"].sort_values()),
            "r--",
            alpha=0.5,
            label=f"Trend: y={z[0]:.2f}x+{z[1]:.2f}",
        )
        plt.legend()

    plt.tight_layout()
    return plt


# ============================================================================
# Visualization 3: Price vs Target Price with Error Bars
# ============================================================================


def viz_price_vs_target(df):
    """Error bar plot showing last price vs target price with confidence intervals."""
    plt.figure(figsize=(14, 7))

    # Prepare data
    df_viz = (
        df[
            [
                "ticker",
                "last_price",
                "price_target_median",
                "price_target_low",
                "price_target_high",
            ]
        ]
        .dropna()
        .head(30)
    )  # Limit to 30 for readability

    # Calculate error bars (distance from median to low/high)
    y_err_lower = df_viz["price_target_median"] - df_viz["price_target_low"]
    y_err_upper = df_viz["price_target_high"] - df_viz["price_target_median"]

    # Create error bar plot
    x_pos = np.arange(len(df_viz))

    plt.errorbar(
        x=x_pos,
        y=df_viz["price_target_median"],
        yerr=[y_err_lower, y_err_upper],
        fmt="o",
        color="darkorange",
        ecolor="gray",
        capsize=4,
        alpha=0.8,
        label="Target Price (with range)",
        markersize=8,
    )

    # Add last price as separate scatter
    plt.scatter(
        x_pos,
        df_viz["last_price"],
        color="blue",
        label="Last Price",
        alpha=0.7,
        s=80,
        marker="s",
    )

    # Formatting
    plt.title(
        "Price vs Target Price (with Confidence Intervals)",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Ticker", fontsize=12)
    plt.ylabel("Price (in USD)", fontsize=12)
    plt.xticks(x_pos, df_viz["ticker"], rotation=90, fontsize=9)
    plt.legend(loc="upper left")
    plt.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()

    return plt


# ============================================================================
# Bonus Visualization 4: ROE vs Debt-to-Equity Ratio
# ============================================================================


def viz_roe_vs_leverage(df):
    """Scatter plot showing ROE vs Debt-to-Equity colored by sector."""
    plt.figure(figsize=(12, 6))

    # Filter valid data
    df_viz = df[["roe", "debt_to_equity", "sector", "ticker"]].dropna()

    # Create scatter plot colored by sector
    sectors = df_viz["sector"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(sectors)))

    for idx, sector in enumerate(sectors):
        sector_data = df_viz[df_viz["sector"] == sector]
        plt.scatter(
            sector_data["debt_to_equity"],
            sector_data["roe"],
            label=sector[:20],  # Truncate long sector names
            alpha=0.7,
            s=100,
            color=colors[idx],
        )

    # Formatting
    plt.title(
        "Return on Equity (ROE) vs Leverage (Debt-to-Equity)",
        fontsize=16,
        fontweight="bold",
    )
    plt.xlabel("Debt-to-Equity Ratio", fontsize=12)
    plt.ylabel("ROE (%)", fontsize=12)
    plt.axhline(y=0, color="white", linestyle="--", alpha=0.3)
    plt.axvline(x=1, color="white", linestyle="--", alpha=0.3)
    plt.grid(linestyle="--", alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    plt.tight_layout()

    return plt


# ============================================================================
# Bonus Visualization 5: Sector Distribution Pie Chart
# ============================================================================


def viz_sector_distribution(df):
    """Pie chart showing distribution of stocks by sector."""
    plt.figure(figsize=(10, 10))

    # Count stocks by sector
    sector_counts = df["sector"].value_counts()

    # Create pie chart with explosion for emphasis
    colors = plt.cm.Set3(np.linspace(0, 1, len(sector_counts)))
    explode = [
        0.05 if i == 0 else 0 for i in range(len(sector_counts))
    ]  # Explode largest sector

    plt.pie(
        sector_counts.values,
        labels=sector_counts.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=colors,
        explode=explode,
        textprops={"fontsize": 10, "color": "white"},
    )

    plt.title("Sector Distribution (First 50 Stocks)", fontsize=16, fontweight="bold")
    plt.axis("equal")
    plt.tight_layout()

    return plt


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Example usage (uncomment when you have the data loaded)
    """
    # Load your data
    all_stocks_preprocessed = pd.read_csv('path/to/data.csv')
    df_50 = all_stocks_preprocessed.head(50)

    # Generate visualizations
    viz_market_cap_by_sector(df_50)
    plt.savefig('viz1_market_cap_by_sector.png', dpi=300, bbox_inches='tight')
    plt.show()

    viz_revenue_vs_ebitda(df_50)
    plt.savefig('viz2_revenue_vs_ebitda.png', dpi=300, bbox_inches='tight')
    plt.show()

    viz_price_vs_target(df_50)
    plt.savefig('viz3_price_vs_target.png', dpi=300, bbox_inches='tight')
    plt.show()

    viz_roe_vs_leverage(df_50)
    plt.savefig('viz4_roe_vs_leverage.png', dpi=300, bbox_inches='tight')
    plt.show()

    viz_sector_distribution(df_50)
    plt.savefig('viz5_sector_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()
    """

    print("Visualization functions defined successfully!")
    print("\nAvailable functions:")
    print("  1. viz_market_cap_by_sector(df)")
    print("  2. viz_revenue_vs_ebitda(df)")
    print("  3. viz_price_vs_target(df)")
    print("  4. viz_roe_vs_leverage(df)")
    print("  5. viz_sector_distribution(df)")
    print("\nUsage:")
    print("  df_50 = all_stocks_preprocessed.head(50)")
    print("  viz_market_cap_by_sector(df_50)")
    print("  plt.show()")
