#!/usr/bin/env python3
"""Train and evaluate CORCD on the six datasets with seeds 2022-2026."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile

from corcd.config import load_config
from corcd.dataset_registry import BENCHMARK_SEEDS, DATASET_IDS

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASET_IDS, default=list(DATASET_IDS))
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--results-root', type=Path, default=ROOT/'results')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if len(args.datasets)!=len(set(args.datasets)):
        parser.error('Do not repeat datasets')
    if args.device!='cpu' and re.fullmatch(r'cuda:[0-9]+',args.device) is None:
        parser.error('Use an explicit CUDA device or cpu')
    data_root=args.data_root.expanduser().resolve()
    results_root=args.results_root.expanduser().resolve()
    print(f'CORCD: {len(args.datasets)} datasets x 5 seeds',flush=True)
    payloads=[]
    for dataset in args.datasets:
        load_config(dataset)
        for seed in BENCHMARK_SEEDS:
            out=results_root/dataset/f'seed_{seed}'
            train=[sys.executable,str(ROOT/'run.py'),'--dataset',dataset,'--seed',str(seed),
                   '--device',args.device,'--data-root',str(data_root),'--output-dir',str(out)]
            if args.force:train.append('--force')
            score=[sys.executable,str(ROOT/'evaluate.py'),'--dataset',dataset,'--seed',str(seed),
                   '--data-root',str(data_root),'--result-dir',str(out)]
            if args.dry_run:
                print(shlex.join(train));print(shlex.join(score))
                continue
            environment=os.environ.copy()
            environment['PYTHONHASHSEED']=str(seed)
            environment.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
            for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
                environment[name]='1'
            environment.setdefault('NUMBA_CACHE_DIR',str(Path(tempfile.gettempdir())/'corcd_v3_numba'))
            environment.setdefault('MPLCONFIGDIR',str(Path(tempfile.gettempdir())/'corcd_v3_mpl'))
            subprocess.run(train,check=True,env=environment)
            subprocess.run(score,check=True,env=environment)
            payloads.append(json.loads((out/'metrics.json').read_text()))
    if args.dry_run:
        return 0
    # Reuse the unchanged evaluator's complete-five-seed summary calculation.
    import evaluate
    for payload in payloads:
        out=results_root/payload['dataset']/f"seed_{payload['seed']}"
        fingerprint=evaluate.validate_run(out,payload['dataset'],payload['seed'])
        if any(payload.get(k)!=v for k,v in fingerprint.items()):
            raise RuntimeError(f'Run changed before summary: {out}')
    summary=evaluate.write_summary(results_root,payloads,args.datasets,evaluate.DEFAULT_METRICS)
    print(f'Summary: {summary}',flush=True)
    return 0


if __name__=='__main__':
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except subprocess.CalledProcessError as error:
        raise SystemExit(error.returncode)
