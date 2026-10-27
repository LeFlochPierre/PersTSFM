"""Univariate, zero-shot Toto 2.0 adapter for PersTSFM.

Uses the installed toto2 API. Run the context-curve runner using the separate
Python 3.12 PERTSFM_TOTO_ENV rather than installing Toto in PERTSFM_ENV.
"""
from __future__ import annotations

import time
import numpy as np
import torch

from src.models.adapters import ForecastModel, ForecastOutput


_CHECKPOINTS = {
    "toto2_22m": "Datadog/Toto-2.0-22m",
    "toto2_313m": "Datadog/Toto-2.0-313m",
}


class Toto2ForecastModel(ForecastModel):
    """ForecastInput -> ForecastOutput interface, with one model load per run."""

    def __init__(self, model_key="toto2_313m"):
        if model_key not in _CHECKPOINTS:
            raise ValueError(f"Unsupported Toto key: {model_key}")
        from toto2 import Toto2Model

        self.name = model_key
        self.model_id = _CHECKPOINTS[model_key]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        start = time.perf_counter()
        self.model = Toto2Model.from_pretrained(self.model_id).to(self.device).eval()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.load_seconds = time.perf_counter() - start
        self.batch_timings = []

    def predict_batch(
        self,
        inputs,
        horizon,
        resolution_minutes,
        context_steps,
        batch_size,
    ):
        del resolution_minutes  # Toto operates on uniformly sampled numeric series.
        horizon, batch_size = int(horizon), int(batch_size)
        if horizon <= 0 or batch_size <= 0:
            raise ValueError("horizon and batch_size must be positive")

        results = []
        for start in range(0, len(inputs), batch_size):
            chunk = inputs[start:start + batch_size]
            vectors = []
            masks = []
            for item in chunk:
                values = np.asarray(item.values, dtype=np.float32).reshape(-1)
                if context_steps is not None:
                    values = values[-int(context_steps):]
                mask = np.isfinite(values)
                if not mask.any():
                    raise ValueError(f"{item.case_id}: no observed history")
                vectors.append(np.where(mask, values, 0.0).astype(np.float32))
                masks.append(mask)

            lengths = {len(v) for v in vectors}
            if len(lengths) != 1:
                raise ValueError("Toto batches require equal-length histories")
            if context_steps is not None and next(iter(lengths)) != int(context_steps):
                raise ValueError("Requested context length does not match provided history")

            # Input: (batch, variates=1, context). Preserve the requested
            # observed context and LEFT-pad only to the Toto patch multiple.
            # Padding is FALSE in the observation mask (not invented history).
            arr = np.stack(vectors, axis=0)[:, None, :]
            obs = np.stack(masks, axis=0)[:, None, :]
            observed_context_steps = int(arr.shape[-1])
            patch_size = int(getattr(getattr(self.model, "config", None), "patch_size", 32))
            if patch_size < 1:
                raise RuntimeError(f"Invalid Toto patch size: {patch_size}")
            padding_steps = (-observed_context_steps) % patch_size
            if padding_steps:
                padding_spec = ((0, 0), (0, 0), (padding_steps, 0))
                arr = np.pad(arr, padding_spec, mode="constant", constant_values=0.0)
                obs = np.pad(obs, padding_spec, mode="constant", constant_values=False)
            assert arr.shape == obs.shape
            assert arr.shape[-1] % patch_size == 0
            inputs_dict = {
                "target": torch.from_numpy(arr).to(self.device),
                "target_mask": torch.from_numpy(obs).to(self.device),
                "series_ids": torch.zeros(len(chunk), 1, dtype=torch.long, device=self.device),
            }

            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
                torch.cuda.reset_peak_memory_stats(self.device)
            start_time = time.perf_counter()
            with torch.inference_mode():
                out = self.model.forecast(
                    inputs_dict,
                    horizon=horizon,
                    decode_block_size=None,
                    has_missing_values=not bool(obs.all()),
                )
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            inference_s = time.perf_counter() - start_time
            memory_bytes = (
                torch.cuda.max_memory_allocated(self.device)
                if self.device.type == "cuda" else 0
            )

            qs = np.asarray(out.detach().float().cpu().numpy(), dtype=np.float64)
            expected = (9, len(chunk), 1, horizon)
            if qs.shape != expected:
                raise RuntimeError(f"Toto quantile shape {qs.shape}; expected {expected}")
            if not np.isfinite(qs).all():
                raise RuntimeError("Toto produced non-finite quantile predictions")

            self.batch_timings.append({
                "model": self.name,
                "checkpoint": self.model_id,
                "context_steps": next(iter(lengths)),
                "padded_context_steps": int(arr.shape[-1]),
                "patch_size": patch_size,
                "left_padding_steps": padding_steps,
                "horizon_steps": horizon,
                "n_cases": len(chunk),
                "inference_seconds": inference_s,
                "peak_cuda_allocated_bytes": int(memory_bytes),
                "has_missing_values": not bool(obs.all()),
            })
            for i, item in enumerate(chunk):
                q10 = qs[0, i, 0, :].copy()
                q50 = qs[4, i, 0, :].copy()
                q90 = qs[8, i, 0, :].copy()
                results.append(ForecastOutput(
                    case_id=item.case_id,
                    prediction=q50.copy(),
                    q10=q10,
                    q50=q50,
                    q90=q90,
                ))
        return results
