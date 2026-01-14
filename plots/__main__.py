"""Allow running as python -m plots."""

from plots.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
