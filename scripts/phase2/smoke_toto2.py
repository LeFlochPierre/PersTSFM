import argparse
import time

import torch
from toto2 import Toto2Model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="Datadog/Toto-2.0-22m",
    )
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required"

    device = torch.device("cuda")
    torch.manual_seed(42)

    print("Model:", args.model_id, flush=True)
    print("Device:", torch.cuda.get_device_name(0), flush=True)

    model = Toto2Model.from_pretrained(args.model_id)
    model = model.to(device).eval()

    # Two deterministic univariate forecasting cases.
    steps = torch.arange(
        192, device=device, dtype=torch.float32
    )

    target = torch.stack([
        torch.sin(steps / 8) + 0.002 * steps,
        0.5 * torch.cos(steps / 12),
    ]).unsqueeze(1)

    assert target.shape == (2, 1, 192)

    target_mask = torch.ones_like(target, dtype=torch.bool)
    series_ids = torch.zeros(
        2, 1, dtype=torch.long, device=device
    )

    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()

    with torch.inference_mode():
        quantiles = model.forecast(
            {
                "target": target,
                "target_mask": target_mask,
                "series_ids": series_ids,
            },
            horizon=12,
            decode_block_size=None,
            has_missing_values=False,
        )

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start

    print("Quantile shape:", tuple(quantiles.shape))

    # Nine quantiles, two cases, one variate, 12 steps.
    assert quantiles.shape == (9, 2, 1, 12)
    assert torch.isfinite(quantiles).all()

    q10 = quantiles[0, :, 0, :]
    q50 = quantiles[4, :, 0, :]
    q90 = quantiles[8, :, 0, :]

    assert q10.shape == q50.shape == q90.shape == (2, 12)
    assert torch.all(q10 <= q50 + 1e-5)
    assert torch.all(q50 <= q90 + 1e-5)

    print("Inference time (s):", round(elapsed, 3))
    print(
        "Peak GPU allocation (GB):",
        round(torch.cuda.max_memory_allocated() / 1e9, 3)
    )
    print("First forecast median:", q50[0].cpu().tolist())
    print("PASS: Toto forecast contract")


if __name__ == "__main__":
    main()
