"""Descriptive paired uncertainty. Never treats correlated sensors as independent datasets.

Entity bootstrap estimates *within-dataset* variability conditional on selected
historical origins. It does NOT provide independent-family generalization CIs.
"""
from __future__ import annotations
import numpy as np


def paired_entity_bootstrap(baseline, candidate, *, replicates=2000, seed=42, alpha=.05):
    """Positive gain means candidate is better. Resample matched entity pairs."""
    base=np.asarray(baseline,dtype=np.float64)
    alt=np.asarray(candidate,dtype=np.float64)
    if (base.ndim!=1 or alt.shape!=base.shape or base.size<2 or
        not np.isfinite(base).all() or not np.isfinite(alt).all() or
        np.mean(base)<=0 or replicates<100 or not 0<alpha<1):
        raise ValueError('Expected >=2 paired finite per-entity errors, valid baseline, >=100 replicates')
    point=100*(np.mean(base)-np.mean(alt))/np.mean(base)
    rng=np.random.default_rng(seed)
    # Chunked to bound RAM for large datasets / bootstrap counts.
    gains=[]
    for start in range(0,replicates,256):
        k=min(256,replicates-start)
        indexes=rng.integers(0,len(base),size=(k,len(base)))
        bm=base[indexes].mean(axis=1)
        am=alt[indexes].mean(axis=1)
        gains.extend((100*(bm-am)/bm).tolist())
    lower,upper=np.quantile(gains,[alpha/2,1-alpha/2])
    return {'n_entities':len(base),'gain_pct':float(point),
            'ci_lower_pct':float(lower),'ci_upper_pct':float(upper),
            'replicates':int(replicates),'alpha':float(alpha),
            'unit':'paired_entity','interpretation':'within_dataset_descriptive_only'}
