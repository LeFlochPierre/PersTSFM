"""Compat entrypoint; use python -m src.budgeted_forecasting.extend_atlas instead."""
from src.budgeted_forecasting.extend_atlas import main

if __name__ == "__main__":
    raise SystemExit(main())
