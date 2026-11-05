"""WP04 from-scratch byte BPE; equivalent to grast train-tokenizer."""

import sys

from llm_scratch.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train-tokenizer", *sys.argv[1:]]))
