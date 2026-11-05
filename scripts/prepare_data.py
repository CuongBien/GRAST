"""WP03 data preparation; equivalent to grast prepare-data."""

import sys

from llm_scratch.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["prepare-data", *sys.argv[1:]]))
