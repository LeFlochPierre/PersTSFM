"""Shared forecasting metrics and forecast shape validation."""
import numpy as np

QUANTILES = np.asarray([.1, .5, .9], dtype=np.float64)

def metrics(pred, target):
    """Metric convention matches Phase 3.3C/D: entity-macro MAE and pinball3."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if (pred.ndim != 3 or pred.shape[-1] != 4 or
        pred.shape[:2] != target.shape or
        not np.isfinite(pred).all() or not np.isfinite(target).all()):
        raise ValueError('Bad prediction/target shape or nonfinite values')
    per_entity = np.abs(pred[...,0]-target).mean(axis=1)
    err = target[...,None]-pred[...,1:]
    pinball = np.maximum(QUANTILES*err,(QUANTILES-1)*err).mean(axis=(1,2))
    return {'mae':float(per_entity.mean()), 'pinball3':float(pinball.mean()),
            'entity_mae':per_entity, 'entity_pinball3':pinball}


def organise_forecast(point, q10, q50, q90, horizon):
    arr = np.column_stack((np.asarray(point).reshape(-1),np.asarray(q10).reshape(-1),
                           np.asarray(q50).reshape(-1),np.asarray(q90).reshape(-1)))
    if arr.shape != (horizon,4) or not np.isfinite(arr).all():
        raise RuntimeError('Forecast has unexpected shape or nonfinite predictions')
    crossed = int(np.any(np.diff(arr[:,1:],axis=1)<-1e-6,axis=1).sum())
    arr[:,1:] = np.sort(arr[:,1:],axis=1)  # identical to previous diagnostic policy
    return arr.astype(np.float32), crossed
