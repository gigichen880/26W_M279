import pandas as pd
import os
import sys

# df_k1 = pd.read_parquet('results/ablation_k/backtest_k1.parquet')
# print("Unique regimes in K=1:", df_k1['regime_assigned'].unique())
# print("Regime prob sum:", df_k1['regime_prob_0'].mean())

# df = pd.read_csv('results/ablation_k_comparison.csv')
# print(df[['K', 'Frobenius', 'LogEuc', 'GMVP_Sharpe', 'Turnover', 'Win_Rate_vs_Roll']])

# df_k1 = pd.read_parquet('results/ablation_k/backtest_k1.parquet')
# # Backtest has one row per date: model_* and roll_* columns (no 'method' column)
# model_sharpe = df_k1['model_gmvp_sharpe'].mean()
# roll_sharpe = df_k1['roll_gmvp_sharpe'].mean()
# print(f"K=1 Model Sharpe: {model_sharpe:.3f}")
# print(f"Roll Sharpe: {roll_sharpe:.3f}")
# print(f"Improvement: {model_sharpe - roll_sharpe:.3f}")

# win_rate = (df_k1['model_fro'] < df_k1['roll_fro']).mean() * 100
# print(f"K=1 win rate vs Roll (Fro): {win_rate:.1f}%")

# Check if K=4 advantage concentrates in crisis periods
df_k1 = pd.read_parquet('results/ablation_k/backtest_k1.parquet')
df_k4 = pd.read_parquet('results/ablation_k/backtest_k4.parquet')

# Ensure date is a column (backtest parquet may have date as index)
for _df in (df_k1, df_k4):
    if 'date' not in _df.columns:
        _df.reset_index(inplace=True)
        if 'date' not in _df.columns and 'index' in _df.columns:
            _df.rename(columns={'index': 'date'}, inplace=True)

# Define crisis periods
crisis_dates = [
    ('2015-08-01', '2016-02-29'),  # China selloff
    ('2018-10-01', '2018-12-31'),  # Q4 selloff
    ('2020-02-20', '2020-04-30'),  # COVID
]

def in_crisis(date):
    for start, end in crisis_dates:
        if start <= str(date)[:10] <= end:
            return True
    return False

df_k1['is_crisis'] = df_k1['date'].apply(in_crisis)
df_k4['is_crisis'] = df_k4['date'].apply(in_crisis)

# One row per date; model_* columns are the model (no 'method' column)
k1_crisis_sharpe = df_k1.loc[df_k1['is_crisis'], 'model_gmvp_sharpe'].mean()
k1_normal_sharpe = df_k1.loc[~df_k1['is_crisis'], 'model_gmvp_sharpe'].mean()
k4_crisis_sharpe = df_k4.loc[df_k4['is_crisis'], 'model_gmvp_sharpe'].mean()
k4_normal_sharpe = df_k4.loc[~df_k4['is_crisis'], 'model_gmvp_sharpe'].mean()

print("Sharpe by period:")
print(f"K=1 Crisis: {k1_crisis_sharpe:.3f}, Normal: {k1_normal_sharpe:.3f}")
print(f"K=4 Crisis: {k4_crisis_sharpe:.3f}, Normal: {k4_normal_sharpe:.3f}")
print(f"K=4 advantage: Crisis: {k4_crisis_sharpe - k1_crisis_sharpe:.3f}, Normal: {k4_normal_sharpe - k1_normal_sharpe:.3f}")

# Full crisis vs normal comparison including Roll (same row has model_* and roll_*)
roll_crisis_sharpe = df_k1.loc[df_k1['is_crisis'], 'roll_gmvp_sharpe'].mean()
roll_normal_sharpe = df_k1.loc[~df_k1['is_crisis'], 'roll_gmvp_sharpe'].mean()

print("\n" + "="*60)
print("CRISIS vs NORMAL PERFORMANCE (Full Comparison)")
print("="*60)
print("\nCRISIS PERIODS (~20% of data):")
print(f"  Roll:  {roll_crisis_sharpe:+.3f}")
print(f"  K=1:   {k1_crisis_sharpe:+.3f}  (vs Roll: {k1_crisis_sharpe - roll_crisis_sharpe:+.3f})")
print(f"  K=4:   {k4_crisis_sharpe:+.3f}  (vs Roll: {k4_crisis_sharpe - roll_crisis_sharpe:+.3f})")

print("\nNORMAL PERIODS (~80% of data):")
print(f"  Roll:  {roll_normal_sharpe:+.3f}")
print(f"  K=1:   {k1_normal_sharpe:+.3f}  (vs Roll: {k1_normal_sharpe - roll_normal_sharpe:+.3f})")
print(f"  K=4:   {k4_normal_sharpe:+.3f}  (vs Roll: {k4_normal_sharpe - roll_normal_sharpe:+.3f})")

print("\nK=4 vs K=1:")
print(f"  Crisis: {k4_crisis_sharpe - k1_crisis_sharpe:+.3f}")
print(f"  Normal: {k4_normal_sharpe - k1_normal_sharpe:+.3f}")