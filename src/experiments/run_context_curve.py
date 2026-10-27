import argparse

from src.data.registry import (
    DATASETS,
)

from src.evaluation.context_curve import (
    run_context_curve,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run a PersTSFM "
            "same-entity context curve."
        )
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=sorted(
            DATASETS
        ),
    )

    parser.add_argument(
        "--model",
        required=True,
        choices=[
            "chronos2",
            "timesfm3",
            "tirex2",
            "toto2_22m",
            "toto2_313m",
            "moirai2_small",
        ],
    )

    parser.add_argument(
        "--split",
        default=None,
    )

    parser.add_argument(
        "--contexts",
        nargs="*",
        type=float,
        default=None,
        help=(
            "Optional context lengths "
            "in hours."
        ),
    )

    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help=(
            "Optional case limit for "
            "smoke testing."
        ),
    )

    parser.add_argument("--sample-entities", type=int, default=None,
                        help="Select a deterministic subset of entities")
    parser.add_argument("--origins-per-entity", type=int, default=None,
                        help="Select temporally spread origins per entity")
    parser.add_argument("--sample-seed", type=int, default=42)

    parser.add_argument(
        "--output-root",
        default="results",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    run_context_curve(
        dataset_key=(
            args.dataset
        ),
        model_key=(
            args.model
        ),
        split=(
            args.split
        ),
        context_hours=(
            args.contexts
        ),
        max_cases=(
            args.max_cases
        ),
        sample_entities=args.sample_entities,
        origins_per_entity=args.origins_per_entity,
        sample_seed=args.sample_seed,
        output_root=(
            args.output_root
        ),
    )


if __name__ == "__main__":
    main()
