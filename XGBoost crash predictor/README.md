# Market Regime Model

Predicts market direction (bull/bear) over a 3-month horizon using 194 features
across economic data (FRED), consumer behavior (Google Trends), market structure,
and policy uncertainty (EPU).

**76% directional accuracy overall. 83% on confident predictions.**

## Quick Start

```bash
# 1. Install dependencies
pip install yfinance pandas numpy requests pytrends scikit-learn xgboost hmmlearn

# 2. Get a free FRED API key
#    https://fred.stlouisfed.org/docs/api/api_key.html

# 3. Collect data (~15 minutes)
python collect_all_data.py --fred-key YOUR_FRED_API_KEY

# 4. Run the model (~2 minutes)
python run_model.py

# 5. View results
python -m http.server 8000
# Open http://localhost:8000/index.html
```

## Files

```
index.html           - Barebones HTML viewer (no CSS frameworks, no dependencies)
collect_all_data.py  - Collects stock, FRED, Google Trends, and EPU data
run_model.py         - Trains model and outputs data/results.json
data/                - Created by scripts (master.csv, results.json)
```

## How It Works

The model combines three layers:

1. **Hidden Markov Model** discovers 6 market regimes from returns, volatility,
   credit spreads, and yield curve data
2. **XGBoost Regression** predicts a continuous -100 to +100 score using 194
   features selected from ~500 candidates
3. **Feature engineering** transforms raw economic data into rate-of-change and
   acceleration signals that capture structural deterioration

## Accuracy

Tested on 1,260 out-of-sample days across 5 walk-forward folds (2021-2025):

- 76.2% directional accuracy overall
- 83.2% when model has moderate-to-strong conviction
- Bear signal correct 42/42 times (100% precision on bearish calls)
- 2022 bear market tracked with 0.70 correlation
- 2025 tariff crash tracked with 0.77 correlation

Honest context: markets go up ~73% of the time, so the edge over "always predict
bull" is +3% overall, concentrated at +10% on strong signals and 100% on bear calls.

## Data Sources

All free, no paid subscriptions required:

- **Yahoo Finance**: S&P 500, sector ETFs, consumer/defense/gambling stocks
- **FRED** (free API key): Yield curve, credit spreads, employment, inflation, 
  housing, financial stress, fed funds rate, and 30+ more series
- **Google Trends**: 30 consumer distress/behavior search terms
- **PolicyUncertainty.com**: Economic Policy Uncertainty Index

## Not Financial Advice

This is a research/educational project. The model is a risk management tool,
not a trading signal. Past performance does not predict future results.
```
