import pandas as pd
import numpy as np

df = pd.read_csv("../outputs/analytics/predictions.csv")

print('=== PREDICTION QUALITY ANALYSIS (8,000 predictions) ===\n')

# Error analysis
print('Error Statistics:')
mae_pct = df['prediction_error_pct'].abs().mean()
median_ae_pct = df['prediction_error_pct'].abs().median()
p90 = df['prediction_error_pct'].abs().quantile(0.90)
p95 = df['prediction_error_pct'].abs().quantile(0.95)
p99 = df['prediction_error_pct'].abs().quantile(0.99)
max_err = df['prediction_error_pct'].abs().max()

print(f'Mean Absolute Error Pct: {mae_pct:.2f}%')
print(f'Median Absolute Error Pct: {median_ae_pct:.2f}%')
print(f'90th Percentile Error: {p90:.2f}%')
print(f'95th Percentile Error: {p95:.2f}%')
print(f'99th Percentile Error: {p99:.2f}%')
print(f'Max Error: {max_err:.2f}%\n')

# Extreme errors
extreme_100 = df[df['prediction_error_pct'].abs() > 100]
extreme_500 = df[df['prediction_error_pct'].abs() > 500]
extreme_1000 = df[df['prediction_error_pct'].abs() > 1000]

print(f'Predictions with >100% error: {len(extreme_100)} ({len(extreme_100)/len(df)*100:.1f}%)')
print(f'Predictions with >500% error: {len(extreme_500)} ({len(extreme_500)/len(df)*100:.1f}%)')
print(f'Predictions with >1000% error: {len(extreme_1000)} ({len(extreme_1000)/len(df)*100:.1f}%)\n')

# Sector performance
print('=== SECTOR PERFORMANCE (sorted by error) ===\n')
sector_stats = df.groupby('sector').agg({
    'prediction_error_pct': lambda x: x.abs().mean(),
    'ticker': 'count'
})
sector_stats.columns = ['mean_abs_error_pct', 'count']
sector_stats = sector_stats.sort_values('mean_abs_error_pct')
print(sector_stats.to_string())

print('\n=== PREDICTION BIAS BY SECTOR ===\n')
bias_stats = df.groupby('sector')['prediction_error'].mean().sort_values()
print(bias_stats.to_string())

print('\n=== UNCERTAINTY QUANTIFICATION ===\n')
df['uncertainty_width'] = df['prediction_upper_90'] - df['prediction_lower_10']
mean_width = df['uncertainty_width'].mean()
median_width = df['uncertainty_width'].median()

print(f'Mean prediction interval width: ${mean_width:.2f}')
print(f'Median prediction interval width: ${median_width:.2f}')

# Check coverage
df['in_interval'] = (df['last_price'] >= df['prediction_lower_10']) & (df['last_price'] <= df['prediction_upper_90'])
coverage = df['in_interval'].sum() / len(df) * 100
print(f'Actual 80% interval coverage: {coverage:.1f}% (target: 80%)\n')

print('=== TOP 10 WORST PREDICTIONS ===\n')
worst = df.nlargest(10, 'prediction_error_pct')[['ticker', 'sector', 'last_price', 'prediction_median', 'prediction_error_pct']].copy()
print(worst.to_string(index=False))

print('\n=== REGION PERFORMANCE ===\n')
region_stats = df.groupby('region').agg({
    'prediction_error_pct': lambda x: x.abs().mean(),
    'ticker': 'count'
})
region_stats.columns = ['mean_abs_error_pct', 'count']
region_stats = region_stats.sort_values('mean_abs_error_pct')
print(region_stats.to_string())

print('\n=== FEATURE AVAILABILITY CHECK ===\n')
feature_cols = ['event_prob_positive', 'event_prob_negative', 'event_prob_neutral', 
                'composite_quality_score', 'momentum_score', 'piotroski_f_score']
for col in feature_cols:
    if col in df.columns:
        non_null = df[col].notna().sum()
        print(f'{col}: {non_null}/{len(df)} ({non_null/len(df)*100:.1f}%)')
