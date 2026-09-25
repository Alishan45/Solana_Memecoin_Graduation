# Solana Memecoin Graduation Prediction
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
## Project Overview
This competition challenges participants to predict whether a newly launched Pump Fun token on Solana will reach 85 SOL in liquidity within the first 100 blocks post-mint. The solution combines **Polars-based parallel processing**, **blockchain-specific feature engineering**, and **gradient-boosted models** to achieve a **log loss of 0.034** on the private leaderboard.

---

## Data Description

### Core Datasets:
1. **train.csv / test_unlabeled.csv**  
   - Token metadata and target labels (`has_graduated`).
   - Key columns: `mint`, `slot_min`, `has_graduated`.

2. **chunk_*.csv**  
   - Transaction data from the first 100 blocks post-mint.  
   - Columns: `block_time`, `slot`, `signing_wallet`, `direction`, `base_coin_amount`, `quote_coin_amount`, etc.

3. **dune_token_info.csv / token_info_onchain_divers.csv**  
   - Token metadata (name, symbol, decimals) and deployment stats (gas used, creator wallet).

---

## Preprocessing Pipeline

### 1. Data Loading & Merging
- **Parallel CSV Loading**:  
  Used `polars` to read 41GB+ of transaction data efficiently:
  ```python
  chunk_files = sorted(glob.glob("/kaggle/input/pump-fun-graduation-february-2025/chunk_*.csv"))
  chunks = [pd.read_csv(f) for f in tqdm(chunk_files)]
  transactions = pd.concat(chunks, ignore_index=True)
  ```
- Time Filtering:
Filtered transactions to the first 100 blocks post-mint:
``` python
transactions_df = transactions_df[transactions_df['slot'] <= transactions_df['slot_min'] + 100]
```
### 2. Missing Value Handling
#### Categorical Imputation:
```python
train[obj_dtypes] = train[obj_dtypes].fillna('missing').astype('category')
```
#### Numerical Imputation:
``` python
train[num_cols] = train[num_cols].fillna(0)
```
### 3. Feature Engineering
#### A. Temporal Features
Cyclical Time Encoding:
``` python
df['created_hour_sin'] = np.sin(2 * np.pi * df['created_hour'] / 24)
df['created_hour_cos'] = np.cos(2 * np.pi * df['created_hour'] / 24)
```
Activity Duration:
``` python
df['activity_duration_sec'] = (df['last_time'] - df['first_time']).dt.total_seconds()
```
#### B. Wallet Behavior
Gini Coefficient (Wallet Concentration):
``` python
sorted_values = wallet_volumes.sort("base_coin_amount").to_pandas()["base_coin_amount"].values
gini_coeff = np.sum((2 * np.arange(1, n+1) - n - 1) * sorted_values) / (n * np.sum(sorted_values))
```
Top-5 Wallet Share:
``` python
top5_wallet_share = np.sum(sorted_values[-5:]) / total_volume
```
#### C. Liquidity Metrics
Virtual SOL Balance (Pool Depth Proxy):
``` python
liq_virtual_sol_balance_after_mean = transactions_df.groupby('base_coin')['virtual_sol_balance_after'].mean()
```
Quote Volume (Total SOL Swapped):
``` python
liq_quote_coin_amount = transactions_df.groupby('base_coin')['quote_coin_amount'].sum()
``` 
#### D. Network Activity
Flow Imbalance (Buy/Sell Pressure):
``` python
flow_imbalance = (buy_quote_amount - sell_quote_amount) / (buy_quote_amount + sell_quote_amount + 1e-6)
```
Gas Efficiency:
``` python
gas_efficiency = quote_sum / (total_consumed_gas + 1e-6)
```
#### E. Token-Specific Features
Name/Symbol Hashing (Dimensionality Reduction):
``` python
df['name_x_hash'] = df['name_x'].apply(lambda x: hash(x) % 500)
```
Meme Token Classification (Regex Patterns):
``` python
df['is_meme_token'] = df['name'].str.contains(r'\b(doge|shiba|floki)\b', case=False, na=False).astype(int)
```
## Model Training
### 1. Class Imbalance Handling
Random Undersampling (40:1 ratio):
``` python
from imblearn.under_sampling import RandomUnderSampler
rus = RandomUnderSampler(sampling_strategy={0: majority_count, 1: minority_count})
X_resampled, y_resampled = rus.fit_resample(X, y)
```
### 2. LightGBM Configuration
Key Hyperparameters:
``` python
lgb_params = {
    'objective': 'binary',
    'metric': 'logloss',
    'learning_rate': 0.01,
    'num_leaves': 31,
    'max_depth': 10,
    'subsample': 0.7,
    'colsample_bytree': 0.7
}
```
### 3. Cross-Validation
Stratified K-Fold:
``` python
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1)
```
## Key Insights
- Early Transaction Velocity (tx_per_sec) was the strongest predictor.
- Creator Reputation (historical graduation rate) had high SHAP values.
- Time-Weighted Features (exponential decay on slot) outperformed simple volume metrics.
## Reproducing Results
- Run Solana_Memecoin_dump_preprocessing.ipynb to generate cleaned data.
- Execute Solana_Memecoin_dump_modeling.ipynb for feature engineering and model training.
- Submit submission.csv with predicted probabilities.
## Dependencies
``` bash
pip install polars lightgbm xgboost scikit-learn imbalanced-learn
```
This solution leverages blockchain-specific behavioral patterns and high-performance data processing to identify tokens likely to graduate versus those prone to rug pulls.
## Author
- [Ali Shan](https://github.com/alishan45)
## License

This project is licensed under the [MIT License](LICENSE).

You are free to use, modify, and distribute this code for personal or commercial purposes.  
**Attribution is required** — please retain the license notice and credit the original author.

© 2024 [[Ali Shan](https://github.com/alishan45)]
