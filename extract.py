"""Convenience entry point matching the brief:  python extract.py invoice.pdf > out.json

Equivalent to `python -m src.main extract ...`; all flags are passed through.
"""

import sys

from src.main import main

if __name__ == "__main__":
    sys.exit(main(["extract", *sys.argv[1:]]))
