from src.selfconfig.execute import (
    add_reference_context_for_legacy_runner,
    context_steps_to_hours,
    group_candidates_by_model,
)


def test_context_steps_to_hours():
    assert (
        context_steps_to_hours(
            96,
            15,
        )
        == 24.0
    )

    assert (
        context_steps_to_hours(
            293,
            5,
        )
        == 293 * 5 / 60
    )


def test_group_candidates_by_model():
    candidates = [
        {
            "model": "chronos2",
            "context_steps": 96,
        },
        {
            "model": "chronos2",
            "context_steps": 192,
        },
        {
            "model": "timesfm3",
            "context_steps": 96,
        },
        {
            "model": "tirex2",
            "context_steps": 768,
        },
    ]

    grouped = (
        group_candidates_by_model(
            candidates
        )
    )

    assert grouped[
        "chronos2"
    ] == [
        96,
        192,
    ]

    assert grouped[
        "timesfm3"
    ] == [
        96,
    ]

    assert grouped[
        "tirex2"
    ] == [
        768,
    ]


def test_reference_context_is_auxiliary_only_when_needed():
    executed, ref, added = (
        add_reference_context_for_legacy_runner(
            [96, 192, 768],
            reference_context_hours=96,
            resolution_minutes=15,
        )
    )

    assert ref == 384
    assert added is True
    assert executed == [
        96,
        192,
        384,
        768,
    ]

    executed, ref, added = (
        add_reference_context_for_legacy_runner(
            [6, 24, 47],
            reference_context_hours=6,
            resolution_minutes=60,
        )
    )

    assert ref == 6
    assert added is False
    assert executed == [
        6,
        24,
        47,
    ]
