"""
=============================================================================
MARKET REGIME MODEL — RUN MODEL & OUTPUT JSON (Finetuned)
=============================================================================
Reads master.csv, runs the finetuned model (HMM + feature interactions +
temporal features + feature-selected XGBoost), outputs results.json.

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
    print("MARKET REGIME MODEL — RUNNING (Finetuned)")
    print("="*60)
    
    if not os.path.exists(DATA_PATH):
        print(f"❌ {DATA_PATH} not found. Run collect_all_data.py first.")
        return
    
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=True)
    print(f"Data: {df.shape[0]} days × {df.shape[1]} columns")
    print(f"Range: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
    
    sp = df['sp500_close']
    vix = df['vix_close']; hyg = df['hyg_close']; tlt = df['tlt_close']; gld = df['gld_close']
    sp_ret = sp.pct_change()
    
    # =================================================================
    # HMM — 6 regimes
    # =================================================================
    print("\n  HMM regime detection...")
    hmm_df = pd.DataFrame(index=df.index)
    hmm_df['r5'] = sp.pct_change(5); hmm_df['r21'] = sp.pct_change(21)
    hmm_df['v21'] = sp_ret.rolling(21).std(); hmm_df['vix'] = vix
    hmm_df['hr'] = hyg.pct_change(21); hmm_df['tr'] = tlt.pct_change(21)
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
        if m.sum() > 0:
            ri.append({'id': r, 'avg_ret': hmm_df['r21'][m].mean(),
                       'n': int(m.sum()), 'avg_vix': hmm_df['vix'][m].mean()})
    ri.sort(key=lambda x: x['avg_ret'])
    rmap = {info['id']: i for i, info in enumerate(ri)}
    rnames = ['Deep crisis', 'Stress', 'Weak', 'Neutral', 'Bullish', 'Strong bull'][:len(ri)]
    ordered = np.array([rmap[r] for r in regime_labels])
    
    current_regime = rnames[ordered[-1]] if len(ordered) > 0 else "Unknown"
    print(f"  Current regime: {current_regime}")
    print(f"  {len(ri)} regimes, {(np.diff(ordered) != 0).sum()} transitions")
    
    reg_df = pd.DataFrame(index=hmm_df.index)
    reg_df['regime'] = ordered
    for i in range(len(ri)): reg_df[f'rp{i}'] = regime_probs[:, ri[i]['id']]
    reg_df['rchg'] = (ordered != np.roll(ordered, 1)).astype(float)
    reg_df['rchg21'] = reg_df['rchg'].rolling(21).sum()
    
    # =================================================================
    # TARGET (balanced)
    # =================================================================
    fwd_ret = pd.Series(index=df.index, dtype=float)
    fwd_dd = pd.Series(index=df.index, dtype=float)
    fwd_gain = pd.Series(index=df.index, dtype=float)
    for i in range(len(sp) - 63):
        c = sp.iloc[i]; f = sp.iloc[i+1:i+64]
        fwd_ret.iloc[i] = (f.iloc[-1]-c)/c
        fwd_dd.iloc[i] = (f.min()-c)/c
        fwd_gain.iloc[i] = (f.max()-c)/c
    
    target = np.clip(0.50*np.clip(fwd_ret.values/0.15*100, -100, 100) +
                     0.25*np.clip(fwd_dd.values/0.30*100, -100, 0) +
                     0.25*np.clip(fwd_gain.values/0.20*100, 0, 100), -100, 100)
    target = pd.Series(target, index=df.index)
    
    # =================================================================
    # FEATURES (finetuned: base + interactions + temporal)
    # =================================================================
    print("  Engineering features...")
    F = pd.DataFrame(index=df.index)
    
    # HMM
    for col in reg_df.columns: F[f'h_{col}'] = reg_df[col]
    
    # Helper: safe get
    def g(col):
        return df[col] if col in df.columns else pd.Series(0, index=df.index)
    
    # FRED rate-of-change
    yc = g('fred_yield_spread_10y2y')
    F['yc_l']=yc; F['yc_r21']=yc.diff(21); F['yc_r63']=yc.diff(63)
    F['yc_acc']=yc.diff(21)-yc.diff(21).shift(21)
    F['yc_inv']=(yc<0).astype(float); F['yc_inv_d']=F['yc_inv'].rolling(63).sum()
    F['yc_steep']=yc.diff(21)*(yc.shift(21)<0).astype(float)
    
    hy = g('fred_ice_bofa_hy_spread')
    F['hy_r5']=hy.diff(5); F['hy_r21']=hy.diff(21); F['hy_r63']=hy.diff(63)
    F['hy_acc']=hy.diff(21)-hy.diff(21).shift(21); F['hy_z']=rz(hy)
    F['baa_r21']=g('fred_credit_spread_baa').diff(21)
    
    fsi = g('fred_financial_stress_stl'); F['fsi']=fsi; F['fsi_r21']=fsi.diff(21)
    
    ff = g('fred_fed_funds_effective'); be5 = g('fred_breakeven_5y')
    F['ff_r63']=ff.diff(63); F['ff_acc']=ff.diff(63)-ff.diff(63).shift(63)
    rr = ff-be5; F['rr_r63']=rr.diff(63); F['rr_r126']=rr.diff(126); F['be_r21']=be5.diff(21)
    
    un = g('fred_unemployment_rate'); F['un_r3m']=un.diff(63)
    F['sahm']=un-un.rolling(126).min()
    pay = g('fred_nonfarm_payrolls')
    F['pay_r3m']=pay.pct_change(63)
    F['pay_acc']=pay.pct_change(63)-pay.pct_change(63).shift(63)
    
    se = g('fred_consumer_sentiment'); F['se_r3m']=se.pct_change(63); F['se_z']=rz(se)
    F['ret_r3m']=g('fred_retail_sales').pct_change(63)
    F['veh_z']=rz(g('fred_vehicle_sales'))
    F['gas_r4w']=g('fred_gasoline_regular').pct_change(21)
    F['oil_r21']=g('fred_oil_wti_daily').pct_change(21)
    F['bl_r3m']=g('fred_bank_loans_total').pct_change(63)
    F['cfnai']=g('fred_cfnai')
    F['hs_r6m']=g('fred_housing_starts').pct_change(126)
    F['mtg_r3m']=g('fred_mortgage_30y').diff(63)
    F['m2_r6m']=g('fred_m2_money').pct_change(126)
    F['ip_r3m']=g('fred_industrial_production').pct_change(63)
    F['cpi_acc']=g('fred_cpi_all').pct_change(63)-g('fred_cpi_all').pct_change(63).shift(63)
    
    # EPU + political
    epu = g('epu_monthly'); F['epu']=epu; F['epu_z']=rz(epu); F['epu_r3m']=epu.pct_change(63)
    F['epu_spike']=(epu>epu.rolling(252).mean()+1.5*epu.rolling(252).std()).astype(float)
    F['epu_x_dd']=F['epu_z']*(-sp.pct_change(21)).clip(lower=0)*10
    if 'equity_uncertainty' in df.columns: F['eq_unc']=df['equity_uncertainty']; F['eq_unc_z']=rz(df['equity_uncertainty'])
    if 'chicago_fci' in df.columns: F['fci']=df['chicago_fci']; F['fci_r21']=df['chicago_fci'].diff(21)
    if 'usd_index' in df.columns: F['usd_r21']=df['usd_index'].pct_change(21); F['usd_z']=rz(df['usd_index'])
    for c in [c for c in df.columns if c.startswith('proxy_')]: F[c.replace('proxy_','px_')]=df[c]
    
    # Market
    for w in [5,21,63,126]: F[f'sp_r{w}']=sp.pct_change(w)
    F['sp_v21']=sp_ret.rolling(21).std(); F['sp_v63']=sp_ret.rolling(63).std()
    F['sp_vr']=F['sp_v21']/F['sp_v63']
    F['sp_dd']=(sp-sp.rolling(252).max())/sp.rolling(252).max()
    F['sp_200']=sp/sp.rolling(200).mean()-1
    F['sp_mac']=sp.rolling(50).mean()/sp.rolling(200).mean()-1
    F['vx_r21']=vix.pct_change(21); F['vx_z']=rz(vix); F['vx_tm']=vix/vix.rolling(63).mean()
    
    scols = [c for c in df.columns if c.startswith('sect_') and c.endswith('_close')]
    for col in scols:
        n = col.replace('_close','')
        F[f'{n}_r21']=df[col].pct_change(21)
        F[f'{n}_dd']=(df[col]-df[col].rolling(252).max())/df[col].rolling(252).max()
    F['sd21']=(df[scols].pct_change(21)<0).sum(axis=1)
    F['sd63']=(df[scols].pct_change(63)<0).sum(axis=1)
    
    for t,l in {'starbucks':'coff','mcdonalds':'ffod','walmart':'wmt','dollar_tree':'dltr',
                'booking':'bkng','airlines':'air','disney':'dis','costco':'cost'}.items():
        if f'{t}_close' in df.columns:
            p=df[f'{t}_close']; F[f'{l}_r21']=p.pct_change(21)
            F[f'{l}_dd']=(p-p.rolling(252).max())/p.rolling(252).max()
    
    if 'sect_disc_close' in df.columns and 'sect_staples_close' in df.columns:
        F['d_vs_s']=(df['sect_disc_close']/df['sect_staples_close']).pct_change(21)
    for t in ['casino_etf','mgm']:
        if f'{t}_close' in df.columns: F[f'{t}_r21']=df[f'{t}_close'].pct_change(21)
    
    dp=[df[f'{t}_close'] for t in ['lockheed','raytheon','northrop'] if f'{t}_close' in df.columns]
    if dp: F['def_sp']=pd.concat(dp,axis=1).mean(axis=1).pct_change(21)-sp.pct_change(21)
    for t in ['homebuilders','realestate','mortgage_reit']:
        if f'{t}_close' in df.columns:
            F[f'{t}_r21']=df[f'{t}_close'].pct_change(21)
            F[f'{t}_dd']=(df[f'{t}_close']-df[f'{t}_close'].rolling(252).max())/df[f'{t}_close'].rolling(252).max()
    if 'btc_close' in df.columns:
        btc=df['btc_close']; F['btc_r21']=btc.pct_change(21)
        F['btc_dd']=(btc-btc.rolling(252).max())/btc.rolling(252).max()
        F['btc_cor']=btc.pct_change().rolling(63).corr(sp.pct_change())
    F['ht_div']=hyg.pct_change(21)-tlt.pct_change(21)
    F['sb_cor']=sp.pct_change().rolling(63).corr(tlt.pct_change())
    if 'copper_close' in df.columns: F['cu_au']=(df['copper_close']/gld).pct_change(21)
    
    for col in [c for c in df.columns if c.startswith('gtrend_')]: F[f'{col}_z']=rz(df[col])
    dz=[F[f'{t}_z'] for t in ['gtrend_recession','gtrend_layoffs','gtrend_bankruptcy','gtrend_foreclosure','gtrend_food_stamps'] if f'{t}_z' in F.columns]
    if dz: F['distr']=pd.concat(dz,axis=1).mean(axis=1)
    fzl=[F[f'{t}_z'] for t in ['gtrend_stock_market_crash','gtrend_bear_market','gtrend_financial_crisis','gtrend_economic_collapse'] if f'{t}_z' in F.columns]
    if fzl: F['fear']=pd.concat(fzl,axis=1).mean(axis=1)
    
    # NEW: Feature interactions (from finetuning)
    print("  Adding finetuned interactions...")
    F['yc_x_hy'] = F['yc_r21'] * F['hy_r21']
    F['yc_inv_x_un'] = F['yc_inv_d'] * F['un_r3m']
    F['vix_x_cr'] = F['vx_z'] * F['hy_z']
    F['se_x_ret'] = F['se_r3m'] * F['ret_r3m']
    F['epu_x_vix'] = F['epu_z'] * F['vx_z']
    F['ff_x_hs'] = F['ff_r63'] * F['hs_r6m']
    F['br_x_vix'] = F['sd63'] * F['vx_z']
    F['gld_x_sp'] = gld.pct_change(21) * (-sp.pct_change(21))
    F['pay_x_se'] = F['pay_acc'] * F['se_r3m']
    
    # NEW: Temporal features (from finetuning)
    F['d_below200'] = (sp < sp.rolling(200).mean()).astype(float).rolling(252).sum()
    F['d_vix20'] = (vix > 20).astype(float).rolling(126).sum()
    F['hy_wid_str'] = (hy.diff(1) > 0).astype(float).rolling(21).sum()
    F['se_dec_str'] = (se.diff(21) < 0).astype(float).rolling(63).sum()
    F['broad_per'] = F['sd63'].rolling(63).mean()
    
    # =================================================================
    # PREPARE + SELECT + TRAIN
    # =================================================================
    F['target'] = target
    fcols = [c for c in F.columns if c != 'target']
    
    # Split: days WITH targets (for training) and ALL days (for prediction)
    # Drop first 252 days (warmup for rolling features)
    F_all = F.iloc[252:]
    F_with_target = F_all.dropna(subset=['target'])
    
    # Build rolling summary for ALL days
    X_raw_all = F_all[fcols].ffill().fillna(0).replace([np.inf, -np.inf], 0)
    
    X_all = pd.DataFrame(index=X_raw_all.index)
    for col in X_raw_all.columns:
        s = X_raw_all[col]
        X_all[f'{col}_c'] = s
        X_all[f'{col}_m'] = s.rolling(252).mean()
        X_all[f'{col}_a'] = s.rolling(21).mean() - s.rolling(252).mean()
    X_all = X_all.ffill().fillna(0).replace([np.inf, -np.inf], 0)
    
    # Training subset (only days with known targets)
    X_train_subset = X_all.loc[F_with_target.index]
    y_train_all = F_with_target['target'].values
    
    # Feature selection (on training data only)
    print(f"  Selecting from {X_all.shape[1]} features...")
    sc_sel = RobustScaler()
    sel = xgb.XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.02,
        subsample=0.7, colsample_bytree=0.4, reg_alpha=3.0, reg_lambda=5.0,
        min_child_weight=30, gamma=0.3, random_state=42, verbosity=0)
    safe_end = max(0, len(X_train_subset) - 252)
    sel.fit(sc_sel.fit_transform(X_train_subset.iloc[:safe_end]), y_train_all[:safe_end])
    imp = pd.Series(sel.feature_importances_, index=X_all.columns)
    nz = imp[imp > 0].sort_values(ascending=False)
    keep = nz.head(max(len(nz)//2, 50)).index.tolist()
    
    # Apply selection to ALL days
    X = X_all[keep]
    X_train_final = X.loc[F_with_target.index]
    y = y_train_all
    print(f"  Selected {len(keep)} features")
    print(f"  Training days: {len(X_train_final)}  |  Total days (incl. recent): {len(X)}")
    
    # Train on all available target data
    print("  Training final model...")
    sc = RobustScaler()
    X_train_scaled = sc.fit_transform(X_train_final)
    
    model = xgb.XGBRegressor(
        n_estimators=800, max_depth=3, learning_rate=0.01,
        subsample=0.7, colsample_bytree=0.4,
        reg_alpha=3.0, reg_lambda=5.0, min_child_weight=30,
        gamma=0.3, random_state=42, verbosity=0
    )
    model.fit(X_train_scaled, y)
    
    # Predict on ALL days (including recent days without targets)
    X_all_scaled = sc.transform(X)
    all_preds = np.clip(model.predict(X_all_scaled), -100, 100)
    
    # Feature importance
    fimp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
    top_features = []
    for f, v in fimp.head(15).items():
        feat_val = X.iloc[-1][f]
        feat_mean = X[f].mean()
        push = "pushing bull" if feat_val > feat_mean else "pushing bear"
        top_features.append({'name': f, 'importance': round(float(v), 4),
                            'value': round(float(feat_val), 4), 'push': push})
    
    # =================================================================
    # BUILD JSON
    # =================================================================
    print("  Building results.json...")
    
    today_score = float(all_preds[-1])
    today_date = str(X.index[-1].strftime('%Y-%m-%d'))
    
    if today_score >= 40: alert = "STRONG BULL"
    elif today_score >= 15: alert = "BULL"
    elif today_score >= 5: alert = "MILD BULL"
    elif today_score >= -5: alert = "NEUTRAL"
    elif today_score >= -15: alert = "MILD BEAR"
    elif today_score >= -40: alert = "BEAR"
    else: alert = "CRISIS"
    
    dates_all = [str(d.strftime('%Y-%m-%d')) for d in X.index]
    scores_all = [round(float(s), 1) for s in all_preds]
    sp500_reindexed = sp.reindex(X.index).ffill()
    sp500_all = [round(float(sp500_reindexed.iloc[i]), 1) if not np.isnan(sp500_reindexed.iloc[i]) else 0 for i in range(len(X))]
    
    # Actual outcomes: available for training days, None for recent days
    target_reindexed = target.reindex(X.index)
    actual_all = []
    for i in range(len(X)):
        val = target_reindexed.iloc[i]
        if pd.notna(val):
            actual_all.append(round(float(val), 1))
        else:
            actual_all.append(None)
    
    regime_hist = []
    reg_reindexed = reg_df.reindex(X.index)['regime'].ffill().fillna(2)
    for i in range(len(X)):
        r = int(reg_reindexed.iloc[i]) if not np.isnan(reg_reindexed.iloc[i]) else 2
        regime_hist.append(rnames[r] if r < len(rnames) else "Unknown")
    
    output = {
        'generated': today_date,
        'current': {
            'date': today_date, 'score': round(today_score, 1),
            'alert': alert, 'regime': current_regime,
            'sp500': round(float(sp.iloc[-1]), 1),
        },
        'top_features': top_features,
        'history': {
            'dates': dates_all, 'scores': scores_all,
            'sp500': sp500_all, 'actual': actual_all, 'regimes': regime_hist,
        },
        'model_info': {
            'features_used': len(keep), 'training_days': len(y),
            'hmm_regimes': len(ri),
            'hmm_transitions': int((np.diff(ordered) != 0).sum()),
            'regime_breakdown': {rnames[i]: ri[i]['n'] for i in range(len(ri))},
            'data_start': df.index.min().strftime('%Y-%m-%d'),
            'data_end': df.index.max().strftime('%Y-%m-%d'),
            'version': 'v8-finetuned',
        },
    }
    
    with open(OUTPUT_PATH, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n  Score: {today_score:+.1f} ({alert})")
    print(f"  Regime: {current_regime}")
    print(f"  S&P 500: {sp.iloc[-1]:.0f}")
    print(f"  History: {len(dates_all)} days")
    print(f"\n✅ Saved {OUTPUT_PATH}")
    print("   Open index.html in your browser to view.")
    print("="*60)

if __name__ == '__main__':
    run()
