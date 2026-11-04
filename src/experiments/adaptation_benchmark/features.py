"""Leakage-free, inexpensive dataset fingerprints from STRICTLY pre-lag2 history.

Features are exploratory covariates, not validated adaptation predictors. Calculated
once per matched entity; used only for analysis, NEVER for lag2 selection in v0.1.
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _correlation(x, lag):
    if lag <= 0 or len(x) <= lag+2:
        return float('nan')
    a,b=x[:-lag],x[lag:]
    sa,sb=np.std(a),np.std(b)
    if sa<=1e-12 or sb<=1e-12:
        return float('nan')
    return float(np.mean(((a-a.mean())/sa)*((b-b.mean())/sb)))


def _spectral_entropy(x):
    x=np.asarray(x,dtype=np.float64)
    if len(x)<8 or np.std(x)<1e-12:
        return float('nan')
    # Linear detrend avoids mistaking nonstationary level drift for a strong frequency.
    time=np.linspace(-1.,1.,len(x))
    detrended=(x-x.mean()) - (np.dot(time,x-x.mean())/np.dot(time,time))*time
    power=np.abs(np.fft.rfft(detrended))**2
    power=power[1:]
    power=power[power>0]
    if len(power)<2:
        return float('nan')
    p=power/power.sum()
    return float(-np.sum(p*np.log(p))/np.log(len(p)))


def compute_features(records, horizon, cadence_minutes):
    """Return per-entity and dataset-median DataFrames; exclude both scored targets."""
    if 1440 % cadence_minutes==0:
        seasonal_lag=1440//cadence_minutes
    else:
        seasonal_lag=7 if cadence_minutes==1440 else max(2,round(1440/cadence_minutes))
    if seasonal_lag==1:
        seasonal_lag=7  # daily frequency -> weekly candidate seasonality
    rows=[]
    for r in records:
        y=np.asarray(r.y[:-2*horizon],dtype=np.float64)
        if len(y)<8 or not np.isfinite(y).all():
            raise ValueError(f'{r.id}: nonfinite/insufficient strictly pre-lag2 feature history')
        scale=float(np.std(y))
        q25,q75=np.percentile(y,[25,75])
        window=max(2,len(y)//4)
        rows.append({
            'entity_id':r.id,
            'train_length':len(y),
            'zero_fraction':float(np.mean(y==0)),
            'scale_std':scale,
            'iqr':float(q75-q25),
            'acf_lag1':_correlation(y,1),
            'acf_candidate_season':_correlation(y,seasonal_lag),
            'season_lag_steps':seasonal_lag,
            'mean_shift_std':float(abs(y[-window:].mean()-y[:window].mean())/max(scale,1e-12)),
            'spectral_entropy':_spectral_entropy(y),
        })
    per_entity=pd.DataFrame(rows)
    med=per_entity.drop(columns=['entity_id']).median(numeric_only=True,skipna=True).to_dict()
    med['entities']=len(per_entity)
    med['horizon']=horizon
    med['cadence_minutes']=cadence_minutes
    return per_entity,pd.DataFrame([med])
