#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import time

from backend.core.llm import get_llm_response


def main() -> int:
    """Load a model once and run one short inference."""
    parser = argparse.ArgumentParser(description='GPU model smoke test')
    parser.add_argument('--model', required=True)
    parser.add_argument(
        '--prompt', default='Кратко представься одним предложением.'
    )
    args = parser.parse_args()

    started = time.perf_counter()
    answer = get_llm_response(args.prompt, model=args.model)
    elapsed = time.perf_counter() - started

    sys.stdout.write(
        f'{json.dumps({"model": args.model, "elapsed_seconds": round(elapsed, 3), "answer_preview": answer[:400]}, ensure_ascii=False, indent=2)}\n'
    )
    return 0
