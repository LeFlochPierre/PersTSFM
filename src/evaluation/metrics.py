import numpy as np


def add_prediction_metrics(pred):
    pred = pred.copy()

    quantiles = pred[
        ["q10", "q50", "q90"]
    ].to_numpy(dtype=float)

    crossing = (
        (quantiles[:, 0] > quantiles[:, 1])
        | (quantiles[:, 1] > quantiles[:, 2])
    )

    sorted_quantiles = np.sort(
        quantiles,
        axis=1,
    )

    pred["q10_eval"] = sorted_quantiles[:, 0]
    pred["q50_eval"] = sorted_quantiles[:, 1]
    pred["q90_eval"] = sorted_quantiles[:, 2]

    pred["quantile_crossing"] = crossing

    pred["error"] = (
        pred["prediction"]
        - pred["truth"]
    )

    pred["abs_error"] = (
        pred["error"].abs()
    )

    pred["squared_error"] = (
        pred["error"] ** 2
    )

    pred["covered_80"] = (
        (pred["truth"] >= pred["q10_eval"])
        & (pred["truth"] <= pred["q90_eval"])
    )

    pred["interval_width_80"] = (
        pred["q90_eval"]
        - pred["q10_eval"]
    )

    pred["negative_prediction"] = (
        pred["prediction"] < 0
    )

    crossing_fraction = float(
        pred["quantile_crossing"].mean()
    )

    return pred, crossing_fraction
