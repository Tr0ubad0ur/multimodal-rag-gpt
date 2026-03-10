#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch


def _run(cmd: list[str]) -> tuple[int, str]:
    try:
        result = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return 127, ''
    return result.returncode, result.stdout.strip()


def main() -> int:
    """Print a compact GPU/server readiness report."""
    payload: dict[str, object] = {
        'python': sys.version.split()[0],
        'platform': sys.platform,
        'cwd': str(Path.cwd()),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'mps_available': bool(
            getattr(torch.backends, 'mps', None)
            and torch.backends.mps.is_available()
        ),
        'gpu_count': torch.cuda.device_count()
        if torch.cuda.is_available()
        else 0,
        'nvidia_smi_present': shutil.which('nvidia-smi') is not None,
    }

    if torch.cuda.is_available():
        payload['devices'] = [
            {
                'index': idx,
                'name': torch.cuda.get_device_name(idx),
                'total_memory_gb': round(
                    torch.cuda.get_device_properties(idx).total_memory
                    / 1024**3,
                    2,
                ),
            }
            for idx in range(torch.cuda.device_count())
        ]

    code, output = _run(
        [
            'nvidia-smi',
            '--query-gpu=name,memory.total,driver_version',
            '--format=csv,noheader',
        ]
    )
    payload['nvidia_smi'] = output.splitlines() if code == 0 else []

    sys.stdout.write(f'{json.dumps(payload, ensure_ascii=False, indent=2)}\n')

    if not payload['cuda_available']:
        return 1
    return 0
