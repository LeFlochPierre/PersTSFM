"""Independent Moirai 2.0 GPU smoke test, before PersTSFM integration."""
import argparse
import platform
import time

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="Salesforce/moirai-2.0-R-small")
    args = parser.parse_args()
    from uni2ts.model.moirai2 import Moirai2Forecast, Moirai2Module

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        raise RuntimeError("GPU smoke test requires CUDA")
    print("Python:", platform.python_version(), flush=True)
    print("Torch:", torch.__version__, flush=True)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("Checkpoint:", args.checkpoint, flush=True)

    t0 = time.perf_counter()
    module = Moirai2Module.from_pretrained(args.checkpoint).to(device).eval()
    torch.cuda.synchronize()
    print("Load seconds:", round(time.perf_counter() - t0, 3), flush=True)

    forecast = Moirai2Forecast(
        module=module, prediction_length=12, context_length=192,
        target_dim=1, feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0,
    ).to(device).eval()

    t = np.arange(192, dtype=np.float32)
    contexts = [np.sin(t / 8) + 0.002 * t, 0.5 * np.cos(t / 12)]
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.inference_mode():
        raw = forecast.predict(contexts)
    torch.cuda.synchronize()
    print("Inference seconds:", round(time.perf_counter() - t0, 3), flush=True)

    out = np.asarray(raw)
    print("Raw shape:", out.shape, flush=True)
    if out.shape == (2, 9, 12, 1):
        out = out[..., 0]
    assert out.shape == (2, 9, 12), out.shape
    assert np.isfinite(out).all()
    print("First forecast median:", out[0, 4, :].tolist(), flush=True)
    print("Quantile crossing fraction:", float(np.mean((out[:, 0] > out[:, 4]) |
                                                     (out[:, 4] > out[:, 8]))), flush=True)
    print("PASS: Moirai forecast contract", flush=True)


if __name__ == "__main__":
    main()
