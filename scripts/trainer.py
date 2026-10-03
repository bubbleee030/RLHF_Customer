#!/usr/bin/env python3
"""VM-friendly entrypoint for the cost model trainer.

This wrapper keeps the command surface simple while reusing the local
`train_cost_model_v2.py` implementation.
"""

from train_cost_model_v2 import main


if __name__ == "__main__":
    raise SystemExit(main())
