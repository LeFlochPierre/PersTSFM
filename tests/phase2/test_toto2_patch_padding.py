"""Verify Toto context padding never introduces observations or future data."""
import numpy as np
import pytest

from test_toto2_adapter import stub, case


@pytest.mark.parametrize("context_steps,expected_total,expected_pad", [
    (6, 32, 26),
    (24, 32, 8),
    (30, 32, 2),
    (32, 32, 0),
    (36, 64, 28),
    (120, 128, 8),
    (180, 192, 12),
])
def test_masked_left_padding(context_steps, expected_total, expected_pad):
    m = stub()
    history = np.arange(1, context_steps + 1, dtype=np.float32)
    result = m.predict_batch(
        [case("sample", history)], horizon=12,
        resolution_minutes=1, context_steps=context_steps, batch_size=1,
    )
    target = m.model.last[0]["target"].numpy()[0, 0]
    mask = m.model.last[0]["target_mask"].numpy()[0, 0]
    assert len(result) == 1
    assert len(target) == len(mask) == expected_total
    np.testing.assert_array_equal(target[-context_steps:], history)
    assert mask[-context_steps:].all()
    assert not mask[:expected_pad].any()
    assert (target[:expected_pad] == 0).all()
    assert m.model.last[3] is (expected_pad > 0)
    timing = m.batch_timings[-1]
    assert timing["context_steps"] == context_steps
    assert timing["padded_context_steps"] == expected_total
    assert timing["left_padding_steps"] == expected_pad
