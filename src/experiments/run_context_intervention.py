import argparse

from src.evaluation.context_intervention import (
    run_context_intervention,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run PersTSFM "
            "context-counterfactual "
            "experiments."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "chronos2",
            "timesfm3",
            "tirex2",
        ],
    )

    parser.add_argument(
        "--recent-hours",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--long-hours",
        required=True,
        type=float,
    )

    parser.add_argument(
        "--split",
        default=None,
    )

    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--one-case-per-entity",
        action="store_true",
        help=(
            "Use only the first evaluation "
            "case from each entity."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_context_intervention(
        dataset_key=(
            args.dataset
        ),
        model_key=(
            args.model
        ),
        recent_hours=(
            args.recent_hours
        ),
        long_hours=(
            args.long_hours
        ),
        split=(
            args.split
        ),
        max_cases=(
            args.max_cases
        ),
        one_case_per_entity=(
            args.one_case_per_entity
        ),
    )


if __name__ == "__main__":
    main()
