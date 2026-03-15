"""
=============================================================================
MARKET REGIME MODEL — DATA COLLECTOR
=============================================================================
Collects all required data and outputs master.csv

SETUP:
    pip install yfinance pandas numpy requests pytrends

USAGE:
    python collect_all_data.py --fred-key YOUR_FRED_API_KEY

OUTPUT:
    ./data/master.csv
=============================================================================
"""

import os, sys, time, argparse
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from io import StringIO

START_DATE = "2015-01-01"
END_DATE = pd.Timestamp.now().strftime("%Y-%m-%d")

os.makedirs("./data/raw", exist_ok=True)

def log(msg): print(f"  {msg}")

# =============================================================================
# 1. STOCK / ETF DATA
# =============================================================================
def collect_stocks():
    print("\n[1/4] STOCK & ETF DATA")
    tickers = {
        "sp500":"^GSPC","vix":"^VIX","hyg":"HYG","tlt":"TLT","gld":"GLD","btc":"BTC-USD",
        "sect_tech":"XLK","sect_health":"XLV","sect_financials":"XLF","sect_disc":"XLY",
        "sect_staples":"XLP","sect_energy":"XLE","sect_industrial":"XLI","sect_materials":"XLB",
        "sect_realestate":"XLRE","sect_utilities":"XLU","sect_comms":"XLC",
        "walmart":"WMT","amazon":"AMZN","costco":"COST","dollar_tree":"DLTR",
        "airlines":"JETS","hotels":"H","marriott":"MAR","booking":"BKNG",
        "disney":"DIS","live_nation":"LYV","starbucks":"SBUX","mcdonalds":"MCD",
        "casino_etf":"BJK","mgm":"MGM","draftkings":"DKNG",
        "philip_morris":"PM","diageo":"DEO",
        "homebuilders":"XHB","realestate":"VNQ","mortgage_reit":"REM",
        "lockheed":"LMT","raytheon":"RTX","northrop":"NOC",
        "copper":"CPER","agriculture":"DBA","lumber":"WOOD","oil_etf":"USO",
    }
    frames = []
    for name, ticker in tickers.items():
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 0:
                df = df[['Open','High','Low','Close','Volume']]
                df.columns = [f"{name}_{c.lower()}" for c in df.columns]
                frames.append(df)
                log(f"✅ {name} ({ticker}): {len(df)} days")
            else:
                log(f"❌ {name}: empty")
        except Exception as e:
            log(f"❌ {name}: {e}")
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()

# =============================================================================
# 2. FRED DATA
# =============================================================================
def collect_fred(api_key):
    print("\n[2/4] FRED ECONOMIC DATA")
    series = {
        "yield_spread_10y2y":"T10Y2Y","yield_spread_10y3m":"T10Y3M",
        "credit_spread_baa":"BAA10Y","ice_bofa_hy_spread":"BAMLH0A0HYM2",
        "vix_fred":"VIXCLS","breakeven_5y":"T5YIE","breakeven_10y":"T10YIE",
        "oil_wti_daily":"DCOILWTICO","treasury_10y":"DGS10","treasury_2y":"DGS2",
        "treasury_3m":"DTB3","fed_funds_effective":"DFF",
        "financial_stress_stl":"STLFSI2",
        "gasoline_regular":"GASREGW","diesel_price":"GASDESW",
        "mortgage_30y":"MORTGAGE30US",
        "bank_loans_total":"TOTLL","consumer_credit_cards":"CCLACBW027SBOG",
        "retail_sales":"RSXFS","pce":"PCE","consumer_sentiment":"UMCSENT",
        "unemployment_rate":"UNRATE","nonfarm_payrolls":"PAYEMS",
        "cpi_all":"CPIAUCSL","core_cpi":"CPILFESL","pce_price":"PCEPI",
        "housing_starts":"HOUST","building_permits":"PERMIT",
        "case_shiller":"CSUSHPINSA",
        "industrial_production":"INDPRO","capacity_utilization":"TCU",
        "durable_goods":"DGORDER","vehicle_sales":"TOTALSA",
        "m2_money":"M2SL","consumer_credit_total":"TOTALSL",
        "trade_balance":"BOPGSTB","air_passengers":"AIRRPMTSI",
        "cfnai":"CFNAI","gdp":"GDP","defense_spending":"FDEFX",
        "cc_delinquency":"DRCCLACBS",
    }
    # Additional political/risk series
    extra = {
        "equity_uncertainty":"WLEMUINDXD","usd_index":"DTWEXBGS",
        "chicago_fci":"NFCI","chicago_leverage":"ANFCI",
    }
    series.update(extra)
    
    frames = []
    for name, sid in series.items():
        try:
            url = (f"https://api.stlouisfed.org/fred/series/observations"
                   f"?series_id={sid}&api_key={api_key}"
                   f"&observation_start={START_DATE}&observation_end={END_DATE}"
                   f"&file_type=json")
            resp = requests.get(url, timeout=30)
            data = resp.json()
            if 'observations' in data and len(data['observations']) > 0:
                df = pd.DataFrame(data['observations'])
                df['date'] = pd.to_datetime(df['date'])
                df['value'] = pd.to_numeric(df['value'], errors='coerce')
                df = df[['date','value']].dropna().set_index('date')
                df.columns = [f"fred_{name}"]
                frames.append(df)
                log(f"✅ fred_{name}: {len(df)} obs")
            else:
                log(f"❌ fred_{name}: no data")
        except Exception as e:
            log(f"❌ fred_{name}: {e}")
        time.sleep(0.3)
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()

# =============================================================================
# 3. GOOGLE TRENDS
# =============================================================================
def collect_trends():
    print("\n[3/4] GOOGLE TRENDS")
    try:
        from pytrends.request import TrendReq
    except ImportError:
        log("❌ pytrends not installed. pip install pytrends")
        return pd.DataFrame()
    
    pytrends = TrendReq(hl='en-US', tz=360, timeout=(10,25))
    all_terms = [
        "recession","layoffs","unemployment benefits","food stamps",
        "bankruptcy","foreclosure","debt consolidation","payday loan",
        "pawn shop","budget grocery",
        "stock market crash","bank run","housing bubble","inflation crisis",
        "fed rate cut","bear market","market correction","financial crisis",
        "stagflation","economic collapse",
        "cancel subscription","cheap flights","used cars","side hustle",
        "coupon codes","discount store","free entertainment","sell my car",
        "how to save money","thrift store",
    ]
    
    frames = []
    for i in range(0, len(all_terms), 5):
        batch = all_terms[i:i+5]
        try:
            pytrends.build_payload(batch, timeframe=f'{START_DATE} {END_DATE}')
            df = pytrends.interest_over_time()
            if len(df) > 0 and 'isPartial' in df.columns:
                df = df.drop('isPartial', axis=1)
                df.columns = [f"gtrend_{t.replace(' ','_')}" for t in batch]
                frames.append(df)
                log(f"✅ {', '.join(batch)}: {len(df)} weeks")
            time.sleep(5)
        except Exception as e:
            log(f"❌ {batch}: {e}")
            time.sleep(30)
    return pd.concat(frames, axis=1) if frames else pd.DataFrame()

# =============================================================================
# 4. EPU
# =============================================================================
def collect_epu():
    print("\n[4/4] ECONOMIC POLICY UNCERTAINTY")
    try:
        url = "https://www.policyuncertainty.com/media/US_Policy_Uncertainty_Data.csv"
        resp = requests.get(url, timeout=30)
        if resp.status_code == 200:
            df = pd.read_csv(StringIO(resp.text))
            if 'Year' in df.columns and 'Month' in df.columns:
                df['date'] = pd.to_datetime(df['Year'].astype(str)+'-'+df['Month'].astype(str)+'-01')
                val_cols = [c for c in df.columns if c not in ['Year','Month','date'] and df[c].dtype in ['float64','int64']]
                if val_cols:
                    df['epu_monthly'] = pd.to_numeric(df[val_cols[0]], errors='coerce')
                    result = df[['date','epu_monthly']].dropna().set_index('date')
                    log(f"✅ EPU monthly: {len(result)} obs")
                    return result
        log("❌ EPU failed")
    except Exception as e:
        log(f"❌ EPU: {e}")
    return pd.DataFrame()

# =============================================================================
# MERGE & SAVE
# =============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fred-key', required=True, help='FRED API key')
    args = parser.parse_args()
    
    print("="*60)
    print("MARKET REGIME MODEL — DATA COLLECTION")
    print("="*60)
    
    stocks = collect_stocks()
    fred = collect_fred(args.fred_key)
    trends = collect_trends()
    epu = collect_epu()
    
    # Align to business days
    bdays = pd.date_range(start=START_DATE, end=END_DATE, freq='B')
    
    print("\n--- MERGING ---")
    master = pd.DataFrame(index=bdays)
    
    for df in [trends, stocks, fred, epu]:
        if len(df) > 0:
            aligned = df.reindex(bdays).ffill()
            for col in aligned.columns:
                master[col] = aligned[col]
    
    master = master.ffill().bfill()
    
    # Compute sentiment proxies
    if 'fred_consumer_sentiment' in master.columns:
        cs = master['fred_consumer_sentiment']
        master['proxy_sentiment_accel'] = cs.pct_change(63) - cs.pct_change(63).shift(63)
        master['proxy_sentiment_crash'] = (cs.pct_change(63) < -0.10).astype(float)
    if 'fred_financial_stress_stl' in master.columns:
        fsi = master['fred_financial_stress_stl']
        master['proxy_fsi_spike'] = (fsi.diff(5) > fsi.diff(5).rolling(252).std() * 2).astype(float)
        master['proxy_fsi_regime'] = (fsi > 0).astype(float).rolling(21).mean()
    if 'fred_ice_bofa_hy_spread' in master.columns:
        hy = master['fred_ice_bofa_hy_spread']
        cv = hy.diff(5) / hy.diff(5).rolling(252).std()
        master['proxy_credit_velocity'] = cv
        master['proxy_credit_shock'] = (cv > 2).astype(float)
    if 'fred_yield_spread_10y2y' in master.columns:
        yc = master['fred_yield_spread_10y2y']
        inv = (yc < 0).astype(float)
        master['proxy_inversion_severity'] = inv.rolling(252).sum() * (-yc).clip(lower=0)
    
    master.to_csv("./data/master.csv")
    print(f"\n✅ Saved ./data/master.csv: {master.shape[0]} rows × {master.shape[1]} columns")
    print("="*60)

if __name__ == '__main__':
    main()
