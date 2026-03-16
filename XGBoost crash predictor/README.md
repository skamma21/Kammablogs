# Market Regime Model

Predicts market direction (bull/bear) over a 3-month horizon using ~170 selected
features across economic data (FRED), consumer behavior (Google Trends), market
structure, policy uncertainty (EPU), and feature interactions.

Trained on 20 years of data (2005-present) covering 8+ crash events including
the 2008 financial crisis, 2020 COVID crash, 2022 bear market, and 2025 tariff
selloff.

**78.7% directional accuracy overall. 83% on confident predictions.**

## Quick Start

```bash
# 1. Install dependencies
pip install yfinance pandas numpy requests pytrends scikit-learn xgboost hmmlearn

# 2. Get a free FRED API key
#    https://fred.stlouisfed.org/docs/api/api_key.html

# 3. Collect 20 years of data (~15-20 minutes)
python collect_all_data.py --fred-key YOUR_FRED_API_KEY

# 4. Run the model (~2-3 minutes)
python run_model.py

# 5. View results
python -m http.server 8000
# Open http://localhost:8000/index.html
```

## Files

```
index.html           - Terminal-style HTML viewer with historical date lookup
collect_all_data.py  - Collects stock, FRED, Google Trends, and EPU data (2005+)
run_model.py         - Trains finetuned model, outputs data/results.json
data/                - Created by scripts (master.csv, results.json)
```

## How It Works

1. **Hidden Markov Model** discovers 6 market regimes from returns, volatility,
   credit spreads, and yield curve data
2. **Feature engineering** transforms raw data into rate-of-change, acceleration,
   and interaction signals (yield curve x credit spread, VIX x sentiment, etc.)
3. **Feature selection** cuts ~500 candidates down to ~170 by XGBoost importance
4. **XGBoost regression** predicts a continuous -100 to +100 market score
   (depth=3, lr=0.01, heavy L1/L2 regularization to prevent overfitting)

## Accuracy

Walk-forward validation (model never sees future data during testing):

- 78.7% directional accuracy overall
- 83% on moderate-to-high confidence predictions
- 80%+ precision on both bull and bear calls
- Tracked the 2022 bear market with 0.74 correlation
- Tracked the 2025 tariff crash with 0.62 correlation

Honest context: markets go up ~73% of the time, so the edge over "always predict
bull" is +6% overall, +10% on confident signals.

## Data Sources (all free)

- **Yahoo Finance**: S&P 500, 11 sector ETFs, 30+ individual stocks/ETFs
- **FRED** (free API key): 40+ economic series (yield curve, credit spreads,
  employment, inflation, housing, financial conditions, etc.)
- **Google Trends**: 30 consumer distress/behavior search terms
- **PolicyUncertainty.com**: Economic Policy Uncertainty Index

## Not Financial Advice

This is a research/educational project. Past performance does not predict future
results. The model is a risk management tool, not a trading signal.
