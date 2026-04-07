"""Allow running the package as `python -m program`."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
