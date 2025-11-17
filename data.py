"""Backward compatible entrypoint that delegates to train.py."""

from train import main


if __name__ == "__main__":
    main()
