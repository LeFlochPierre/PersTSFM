"""Zero-shot, target-only Moirai 2.0 adapter for PersTSFM.

Independent model environment recommended. The official uni2ts dependency metadata
pins torch<2.5, so validate separately on recent CUDA/Blackwell builds.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from src.models.adapters import ForecastModel, ForecastOutput


class Moirai2ForecastModel(ForecastModel):
    """Match the existing ForecastInput/ForecastOutput contract.

    The expensive pretrained module is loaded once. The lightweight forecasting
    wrapper is rebuilt only when context length or prediction horizon changes.
    """

    name = "moirai2_small"

    def __init__(self, model_id: str = "Salesforce/moirai-2.0-R-small"):
        from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

        self.model_id = model_id
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._forecast_cls = Moirai2Forecast
        t0 = time.perf_counter()
        self.module = Moirai2Module.from_pretrained(model_id).to(self.device).eval()
        self._sync()
        self.load_seconds = time.perf_counter() - t0
        self._forecast = None
        self._shape = None
        self.batch_timings: list[dict[str, Any]] = []

    def _sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _configure(self, context_steps: int, horizon: int) -> float:
        shape = (context_steps, horizon)
        if shape == self._shape:
            return 0.0
        self._sync()
        t0 = time.perf_counter()
        self._forecast = self._forecast_cls(
            module=self.module,
            prediction_length=horizon,
            context_length=context_steps,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
        ).to(self.device).eval()
        self._sync()
        self._shape = shape
        return time.perf_counter() - t0

    def predict_batch(
        self,
        inputs,
        horizon,
        resolution_minutes,
        context_steps,
        batch_size,
    ):
        del resolution_minutes
        if not inputs:
            return []
        horizon = int(horizon)
        batch_size = int(batch_size)
        context_steps = int(context_steps)
        if horizon <= 0 or batch_size <= 0 or context_steps <= 0:
            raise ValueError("horizon, context_steps, and batch_size must be positive")

        configure_s = self._configure(context_steps, horizon)
        outputs = []
        for start in range(0, len(inputs), batch_size):
            chunk = inputs[start:start + batch_size]
            values = []
            for item in chunk:
                vector = np.asarray(item.values, dtype=np.float32).reshape(-1)
                if len(vector) != context_steps:
                    raise ValueError(
                        f"{item.case_id}: context length mismatch: "
                        f"expected {context_steps}, got {len(vector)}"
                    )
                if not np.isfinite(vector).any():
                    raise ValueError(f"{item.case_id}: no observed history")
                if np.isinf(vector).any():
                    raise ValueError(f"{item.case_id}: infinite history values")
                # Moirai2Forecast.predict handles NaNs with its own causal
                # imputation; no future targets/covariates are supplied.
                values.append(vector)

            self._sync()
            if self.device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(self.device)
            t0 = time.perf_counter()
            with torch.inference_mode():
                raw = self._forecast.predict(values)
            self._sync()
            elapsed_s = time.perf_counter() - t0
            peak_bytes = (
                int(torch.cuda.max_memory_allocated(self.device))
                if self.device.type == "cuda" else 0
            )

            qs = np.asarray(raw, dtype=np.float64)
            if qs.shape == (len(chunk), 9, horizon, 1):
                qs = qs[..., 0]
            expected = (len(chunk), 9, horizon)
            if qs.shape != expected:
                raise RuntimeError(
                    f"Moirai output shape {qs.shape}, expected {expected}"
                )
            if not np.isfinite(qs).all():
                raise RuntimeError("Moirai output contains non-finite quantiles")

            self.batch_timings.append({
                "model": self.name,
                "checkpoint": self.model_id,
                "context_steps": context_steps,
                "horizon_steps": horizon,
                "n_cases": len(chunk),
                "inference_seconds": elapsed_s,
                "configure_seconds": configure_s,
                "peak_cuda_allocated_bytes": peak_bytes,
                "has_missing_values": any(np.isnan(v).any() for v in values),
            })
            configure_s = 0.0
            for i, item in enumerate(chunk):
                q10, q50, q90 = (
                    qs[i, 0, :].copy(),
                    qs[i, 4, :].copy(),
                    qs[i, 8, :].copy(),
                )
                outputs.append(ForecastOutput(
                    case_id=item.case_id,
                    prediction=q50.copy(),
                    q10=q10,
                    q50=q50,
                    q90=q90,
                ))
        return outputs
