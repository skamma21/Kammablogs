"""
=============================================================================
MARKET REGIME MODEL — RUN MODEL & OUTPUT JSON
=============================================================================
Reads master.csv, runs the V8 model (HMM + feature-selected XGBoost),
outputs results.json for the HTML viewer.

USAGE:
    python run_model.py

INPUT:  ./data/master.csv
OUTPUT: ./data/results.json
=============================================================================
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.ensemble import IsolationForest
from hmmlearn.hmm import GaussianHMM
import xgboost as xgb
import json, warnings, os
warnings.filterwarnings('ignore')
np.random.seed(42)

DATA_PATH = "./data/master.csv"
OUTPUT_PATH = "./data/results.json"

def rz(s, w=252):
    return (s - s.rolling(w).mean()) / s.rolling(w).std()

def run():
    print("="*60)
    print("MARKET REGIME MODEL — RUNNING")
    print("="*60)
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ {DATA_PATH} not found. Run collect_all_data.py first.")
        return
    
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    print(f"Data: {df.shape[0]} days × {df.shape[1]} columns")
    
    sp = df['sp500_close']
    vix = df['vix_close']; hyg = df['hyg_close']; tlt = df['tlt_close']; gld = df['gld_close']
    sp_ret = sp.pct_change()
    
    # =================================================================
    # HMM
    # =================================================================
    print("\n  HMM regime detection...")
    hmm_df = pd.DataFrame(index=df.index)
    hmm_df['ret_5d'] = sp.pct_change(5); hmm_df['ret_21d'] = sp.pct_change(21)
    hmm_df['vol_21d'] = sp_ret.rolling(21).std(); hmm_df['vix'] = vix
    hmm_df['hyg_ret'] = hyg.pct_change(21); hmm_df['tlt_ret'] = tlt.pct_change(21)
    if 'fred_yield_spread_10y2y' in df.columns: hmm_df['yc'] = df['fred_yield_spread_10y2y']
    if 'fred_ice_bofa_hy_spread' in df.columns: hmm_df['hy'] = df['fred_ice_bofa_hy_spread']
    hmm_df = hmm_df.dropna()
    
    best_hmm = None; best_s = -np.inf
    for _ in range(5):
        try:
            h = GaussianHMM(n_components=6, covariance_type='diag', n_iter=300,
                            random_state=np.random.randint(10000))
            h.fit(StandardScaler().fit_transform(hmm_df))
            s = h.score(StandardScaler().fit_transform(hmm_df))
            if s > best_s: best_s = s; best_hmm = h
        except: pass
    
    hmm_scaled = StandardScaler().fit_transform(hmm_df)
    regime_probs = best_hmm.predict_proba(hmm_scaled)
    regime_labels = best_hmm.predict(hmm_scaled)
    
    ri = []
    for r in range(6):
        m = regime_labels == r
        if m.sum() == 0: continue
        ri.append({'id':r,'avg_ret':hmm_df['ret_21d'][m].mean(),'n':int(m.sum()),
                   'avg_vix':hmm_df['vix'][m].mean()})
    ri.sort(key=lambda x: x['avg_ret'])
    rmap = {info['id']:i for i,info in enumerate(ri)}
    rnames = ['Deep crisis','Stress','Weak','Neutral','Bullish','Strong bull'][:len(ri)]
    ordered = np.array([rmap[r] for r in regime_labels])
    
    reg_df = pd.DataFrame(index=hmm_df.index)
    reg_df['regime'] = ordered
    for i in range(len(ri)): reg_df[f'reg_{i}_p'] = regime_probs[:,ri[i]['id']]
    reg_df['reg_changed'] = (ordered != np.roll(ordered,1)).astype(float)
    reg_df['reg_changes_21d'] = reg_df['reg_changed'].rolling(21).sum()
    
    current_regime = rnames[ordered[-1]] if len(ordered) > 0 else "Unknown"
    print(f"  Current regime: {current_regime}")
    
    # =================================================================
    # TARGET
    # =================================================================
    fwd_ret = pd.Series(index=df.index, dtype=float)
    fwd_dd = pd.Series(index=df.index, dtype=float)
    fwd_gain = pd.Series(index=df.index, dtype=float)
    for i in range(len(sp)-63):
        c = sp.iloc[i]; f = sp.iloc[i+1:i+64]
        fwd_ret.iloc[i] = (f.iloc[-1]-c)/c; fwd_dd.iloc[i] = (f.min()-c)/c
        fwd_gain.iloc[i] = (f.max()-c)/c
    
    target = np.clip(0.50*np.clip(fwd_ret.values/0.15*100,-100,100) +
                     0.25*np.clip(fwd_dd.values/0.30*100,-100,0) +
                     0.25*np.clip(fwd_gain.values/0.20*100,0,100), -100, 100)
    target = pd.Series(target, index=df.index)
    
    # =================================================================
    # FEATURES (same as V8)
    # =================================================================
    print("  Engineering features...")
    features = pd.DataFrame(index=df.index)
    
    for col in reg_df.columns: features[f'h_{col}'] = reg_df[col]
    
    # New political data
    for col_name, df_col in [('eq_unc','equity_uncertainty'),('usd_roc_21d','usd_index'),
                              ('fci','chicago_fci'),('fci_adj','chicago_leverage')]:
        if df_col in df.columns:
            s = df[df_col]
            if 'roc' in col_name: features[col_name] = s.pct_change(21)
            else: features[col_name] = s
            features[f'{col_name}_z'] = rz(s)
    
    if 'epu_monthly' in df.columns:
        epu = df['epu_monthly']
        features['epu'] = epu; features['epu_z'] = rz(epu)
        features['epu_roc_3m'] = epu.pct_change(63)
        features['epu_spike'] = (epu > epu.rolling(252).mean()+1.5*epu.rolling(252).std()).astype(float)
        features['epu_x_dd'] = features['epu_z']*(-sp.pct_change(21)).clip(lower=0)*10
    
    for c in [c for c in df.columns if c.startswith('proxy_')]:
        features[c.replace('proxy_','px_')] = df[c]
    
    # FRED
    yc = df.get('fred_yield_spread_10y2y')
    if yc is not None:
        features['yc_lev'] = yc; features['yc_roc21'] = yc.diff(21); features['yc_roc63'] = yc.diff(63)
        features['yc_acc'] = yc.diff(21)-yc.diff(21).shift(21)
        features['yc_inv'] = (yc<0).astype(float); features['yc_inv_d'] = features['yc_inv'].rolling(63).sum()
        features['yc_steep'] = yc.diff(21)*(yc.shift(21)<0).astype(float)
    
    hy = df.get('fred_ice_bofa_hy_spread')
    if hy is not None:
        features['hy_r5']=hy.diff(5); features['hy_r21']=hy.diff(21); features['hy_r63']=hy.diff(63)
        features['hy_acc']=hy.diff(21)-hy.diff(21).shift(21); features['hy_z']=rz(hy)
    
    baa = df.get('fred_credit_spread_baa')
    if baa is not None: features['baa_r21'] = baa.diff(21)
    
    fsi = df.get('fred_financial_stress_stl')
    if fsi is not None: features['fsi_l']=fsi; features['fsi_r21']=fsi.diff(21)
    
    ff = df.get('fred_fed_funds_effective'); be5 = df.get('fred_breakeven_5y')
    if ff is not None:
        features['ff_r63']=ff.diff(63); features['ff_acc']=ff.diff(63)-ff.diff(63).shift(63)
    if ff is not None and be5 is not None:
        rr=ff-be5; features['rr_r63']=rr.diff(63); features['rr_r126']=rr.diff(126)
        features['be_r21']=be5.diff(21)
    
    un = df.get('fred_unemployment_rate')
    if un is not None:
        features['un_r3m']=un.diff(63); features['sahm']=un-un.rolling(126).min()
    pay = df.get('fred_nonfarm_payrolls')
    if pay is not None:
        features['pay_r3m']=pay.pct_change(63)
        features['pay_acc']=pay.pct_change(63)-pay.pct_change(63).shift(63)
    
    se = df.get('fred_consumer_sentiment')
    if se is not None: features['se_r3m']=se.pct_change(63); features['se_z']=rz(se)
    
    ret_s = df.get('fred_retail_sales')
    if ret_s is not None: features['ret_r3m']=ret_s.pct_change(63)
    
    vs = df.get('fred_vehicle_sales')
    if vs is not None: features['veh_z']=rz(vs)
    
    gas = df.get('fred_gasoline_regular')
    if gas is not None: features['gas_r4w']=gas.pct_change(21)
    oil = df.get('fred_oil_wti_daily')
    if oil is not None: features['oil_r21']=oil.pct_change(21)
    bl = df.get('fred_bank_loans_total')
    if bl is not None: features['bl_r3m']=bl.pct_change(63)
    cfn = df.get('fred_cfnai')
    if cfn is not None: features['cfnai']=cfn
    hs = df.get('fred_housing_starts')
    if hs is not None: features['hs_r6m']=hs.pct_change(126)
    mtg = df.get('fred_mortgage_30y')
    if mtg is not None: features['mtg_r3m']=mtg.diff(63)
    m2 = df.get('fred_m2_money')
    if m2 is not None: features['m2_r6m']=m2.pct_change(126)
    ip = df.get('fred_industrial_production')
    if ip is not None: features['ip_r3m']=ip.pct_change(63)
    cpi = df.get('fred_cpi_all')
    if cpi is not None: features['cpi_acc']=cpi.pct_change(63)-cpi.pct_change(63).shift(63)
    
    # Market
    for w in [5,21,63,126]: features[f'sp_r{w}']=sp.pct_change(w)
    features['sp_v21']=sp_ret.rolling(21).std(); features['sp_v63']=sp_ret.rolling(63).std()
    features['sp_vr']=features['sp_v21']/features['sp_v63']
    features['sp_dd']=(sp-sp.rolling(252).max())/sp.rolling(252).max()
    features['sp_200']=sp/sp.rolling(200).mean()-1
    features['sp_mac']=sp.rolling(50).mean()/sp.rolling(200).mean()-1
    features['vx_r21']=vix.pct_change(21); features['vx_z']=rz(vix)
    features['vx_tm']=vix/vix.rolling(63).mean()
    
    scols = [c for c in df.columns if c.startswith('sect_') and c.endswith('_close')]
    for col in scols:
        n=col.replace('_close','')
        features[f'{n}_r21']=df[col].pct_change(21)
        features[f'{n}_dd']=(df[col]-df[col].rolling(252).max())/df[col].rolling(252).max()
    features['sd21']=(df[scols].pct_change(21)<0).sum(axis=1)
    features['sd63']=(df[scols].pct_change(63)<0).sum(axis=1)
    
    for t,l in {'starbucks':'coff','mcdonalds':'ffod','walmart':'wmt','dollar_tree':'dltr',
                'booking':'bkng','airlines':'air','disney':'dis','costco':'cost'}.items():
        if f'{t}_close' in df.columns:
            p=df[f'{t}_close']
            features[f'{l}_r21']=p.pct_change(21)
            features[f'{l}_dd']=(p-p.rolling(252).max())/p.rolling(252).max()
    
    if 'sect_disc_close' in df.columns and 'sect_staples_close' in df.columns:
        features['d_vs_s']=(df['sect_disc_close']/df['sect_staples_close']).pct_change(21)
    for t in ['casino_etf','mgm']:
        if f'{t}_close' in df.columns: features[f'{t}_r21']=df[f'{t}_close'].pct_change(21)
    
    dp = [df[f'{t}_close'] for t in ['lockheed','raytheon','northrop'] if f'{t}_close' in df.columns]
    if dp: features['def_sp']=pd.concat(dp,axis=1).mean(axis=1).pct_change(21)-sp.pct_change(21)
    
    for t in ['homebuilders','realestate','mortgage_reit']:
        if f'{t}_close' in df.columns:
            features[f'{t}_r21']=df[f'{t}_close'].pct_change(21)
            features[f'{t}_dd']=(df[f'{t}_close']-df[f'{t}_close'].rolling(252).max())/df[f'{t}_close'].rolling(252).max()
    
    if 'btc_close' in df.columns:
        btc=df['btc_close']
        features['btc_r21']=btc.pct_change(21)
        features['btc_dd']=(btc-btc.rolling(252).max())/btc.rolling(252).max()
        features['btc_cor']=btc.pct_change().rolling(63).corr(sp.pct_change())
    
    features['ht_div']=hyg.pct_change(21)-tlt.pct_change(21)
    features['sb_cor']=sp.pct_change().rolling(63).corr(tlt.pct_change())
    if 'copper_close' in df.columns: features['cu_au']=(df['copper_close']/gld).pct_change(21)
    
    for col in [c for c in df.columns if c.startswith('gtrend_')]:
        features[f'{col}_z']=rz(df[col])
    
    dz=[features[f'{t}_z'] for t in ['gtrend_recession','gtrend_layoffs','gtrend_bankruptcy','gtrend_foreclosure','gtrend_food_stamps'] if f'{t}_z' in features.columns]
    if dz: features['distr']=pd.concat(dz,axis=1).mean(axis=1)
    fzl=[features[f'{t}_z'] for t in ['gtrend_stock_market_crash','gtrend_bear_market','gtrend_financial_crisis','gtrend_economic_collapse'] if f'{t}_z' in features.columns]
    if fzl: features['fear']=pd.concat(fzl,axis=1).mean(axis=1)
    
    # =================================================================
    # PREPARE + SELECT FEATURES
    # =================================================================
    features['target'] = target
    fcols = [c for c in features.columns if c != 'target']
    fc = features.dropna(subset=['target']).iloc[252:]
    X_raw = fc[fcols].ffill().fillna(0).replace([np.inf,-np.inf],0)
    y = fc['target'].values
    
    X = pd.DataFrame(index=X_raw.index)
    for col in X_raw.columns:
        s = X_raw[col]
        X[f'{col}_c'] = s
        X[f'{col}_m'] = s.rolling(252).mean()
        X[f'{col}_a'] = s.rolling(21).mean()-s.rolling(252).mean()
    X = X.ffill().fillna(0).replace([np.inf,-np.inf],0)
    
    # Feature selection
    print("  Selecting features...")
    sc_sel = RobustScaler()
    sel_model = xgb.XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.02,
        subsample=0.7, colsample_bytree=0.4, reg_alpha=3.0, reg_lambda=5.0,
        min_child_weight=30, gamma=0.3, random_state=42, verbosity=0)
    sel_model.fit(sc_sel.fit_transform(X.iloc[:-252]), y[:-252])
    imp = pd.Series(sel_model.feature_importances_, index=X.columns)
    nz = imp[imp>0].sort_values(ascending=False)
    keep = nz.head(max(len(nz)//2, 50)).index.tolist()
    X = X[keep]
    print(f"  Selected {len(keep)} features")
    
    # =================================================================
    # TRAIN FINAL MODEL ON ALL DATA EXCEPT LAST 63 DAYS
    # =================================================================
    print("  Training final model...")
    sc = RobustScaler()
    X_train = sc.fit_transform(X.iloc[:-63])
    X_latest = sc.transform(X.iloc[-63:])
    y_train = y[:-63]
    
    model = xgb.XGBRegressor(
        n_estimators=800, max_depth=3, learning_rate=0.01,
        subsample=0.7, colsample_bytree=0.4,
        reg_alpha=3.0, reg_lambda=5.0, min_child_weight=30,
        gamma=0.3, random_state=42, verbosity=0
    )
    model.fit(X_train, y_train)
    
    # Predict on full history for display
    X_all_scaled = sc.transform(X)
    all_preds = np.clip(model.predict(X_all_scaled), -100, 100)
    
    # Feature importance
    fimp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_features = []
    for f, v in fimp.head(15).items():
        direction = "bear" if X.iloc[-1][f] < X.iloc[-1][f] else "neutral"
        # Determine if feature is pushing bull or bear today
        feat_val = X.iloc[-1][f]
        feat_mean = X[f].mean()
        if feat_val > feat_mean:
            push = "pushing bull" if fimp[f] > 0 else "pushing bear"
        else:
            push = "pushing bear" if fimp[f] > 0 else "pushing bull"
        top_features.append({
            'name': f,
            'importance': round(float(v), 4),
            'value': round(float(feat_val), 4),
            'push': push,
        })
    
    # =================================================================
    # BUILD OUTPUT JSON
    # =================================================================
    print("  Building results.json...")
    
    today_score = float(all_preds[-1])
    today_date = str(X.index[-1].strftime('%Y-%m-%d'))
    
    # Alert level
    if today_score >= 40: alert = "STRONG BULL"
    elif today_score >= 15: alert = "BULL"
    elif today_score >= 5: alert = "MILD BULL"
    elif today_score >= -5: alert = "NEUTRAL"
    elif today_score >= -15: alert = "MILD BEAR"
    elif today_score >= -40: alert = "BEAR"
    else: alert = "CRISIS"
    
    # History (daily for last 6 months, weekly before that)
    dates_all = [str(d.strftime('%Y-%m-%d')) for d in X.index]
    scores_all = [round(float(s), 1) for s in all_preds]
    sp500_all = [round(float(sp.reindex(X.index).iloc[i]), 1) if not np.isnan(sp.reindex(X.index).iloc[i]) else 0 for i in range(len(X))]
    actual_all = [round(float(y[i]), 1) for i in range(len(y))]
    
    # Regime history
    regime_hist = []
    reg_reindexed = reg_df.reindex(X.index)['regime'].ffill().fillna(2)
    for i in range(len(X)):
        r = int(reg_reindexed.iloc[i]) if not np.isnan(reg_reindexed.iloc[i]) else 2
        regime_hist.append(rnames[r] if r < len(rnames) else "Unknown")
    
    # Export FULL daily history so the HTML viewer can jump to any date
    output = {
        'generated': today_date,
        'current': {
            'date': today_date,
            'score': round(today_score, 1),
            'alert': alert,
            'regime': current_regime,
            'sp500': round(float(sp.iloc[-1]), 1),
        },
        'top_features': top_features,
        'history': {
            'dates': dates_all,
            'scores': scores_all,
            'sp500': sp500_all,
            'actual': actual_all,
            'regimes': regime_hist,
        },
        'model_info': {
            'features_used': len(keep),
            'training_days': len(y_train),
            'hmm_regimes': len(ri),
            'hmm_transitions': int((np.diff(ordered)!=0).sum()),
            'regime_breakdown': {rnames[i]: ri[i]['n'] for i in range(len(ri))},
        },
    }
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  Score: {today_score:+.1f} ({alert})")
    print(f"  Regime: {current_regime}")
    print(f"  S&P 500: {sp.iloc[-1]:.0f}")
    print(f"\n✅ Saved {OUTPUT_PATH}")
    print("   Open index.html in your browser to view.")
    print("="*60)

if __name__ == '__main__':
    run()
