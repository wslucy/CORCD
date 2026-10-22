#!/usr/bin/env python3
"""Train CORCD on one dataset and random seed."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from corcd.config import BENCHMARK_SEEDS, DATASET_IDS, config_hash, load_config
from corcd.runtime import ROOT, completed, input_signature, set_seed, source_hash, write_info


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--dataset", required=True, choices=DATASET_IDS)
    result.add_argument("--seed", type=int, default=2022, choices=BENCHMARK_SEEDS)
    result.add_argument("--data-root", type=Path, default=ROOT / "data")
    result.add_argument("--output-dir", type=Path)
    result.add_argument("--device", default="cuda:0", help="Explicit PyTorch device; CPU is available for smoke tests.")
    result.add_argument("--force", action="store_true", help="Recompute a completed run.")
    result.add_argument("--dry-run", action="store_true", help="Print the resolved configuration without training.")
    return result


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    config = load_config(args.dataset)
    config_id = config_hash(config)
    data_root = args.data_root.expanduser().resolve()
    out = (args.output_dir or ROOT / "single_results" / args.dataset / f"seed_{args.seed}").expanduser().resolve()
    if args.dry_run:
        print(json.dumps({"dataset": args.dataset, "seed": args.seed, "device": args.device,
                          "data_root": str(data_root), "output_dir": str(out),
                          "config_id": config_id, "config": config}, indent=2))
        return
    if not args.force and completed(out, dataset=args.dataset, seed=args.seed, config_id=config_id,
                                    device=args.device, data_root=data_root):
        print(f"[SKIP] {out}")
        return

    out.mkdir(parents=True, exist_ok=True)
    info = {
        "method": "CORCD", "dataset": args.dataset, "seed": args.seed,
        "status": "running",
        "config_id": config_id, "source_id": source_hash(),
        "device": args.device, "data_root": str(data_root),
    }
    write_info(out, info)
    # A rerun must invalidate scores produced from earlier predictions.
    for name in ("metrics.json", "metrics.csv"):
        (out / name).unlink(missing_ok=True)
    if out.name == f"seed_{args.seed}" and out.parent.name == args.dataset:
        summary = out.parent.parent / "summary"
        for name in ("all_runs.csv", "summary_long.csv", "summary_wide.csv"):
            (summary / name).unlink(missing_ok=True)
    try:
        # Set process-level flags before importing PyTorch or constructing a model.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS"):
            os.environ.setdefault(name, "1")
        os.environ.setdefault("NUMBA_CACHE_DIR", str(Path(tempfile.gettempdir()) / "corcd_v3_numba"))
        os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "corcd_v3_mpl"))
        import numpy as np
        import pandas as pd
        import torch
        import yaml
        from corcd.clustering import clustering
        from corcd.data import load_dataset
        from corcd.graphs import build_multiscale_bundle
        from corcd.preprocessing import CANONICAL_MODALITY_ORDER, prepare_benchmark_data
        from corcd.trainer import train_corcd

        device = torch.device(args.device)
        if device.type not in {"cuda", "cpu"}:
            raise ValueError("Choose an explicit CUDA device or cpu.")
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable. Check the PyTorch/CUDA environment.")
        info.update(
            device=str(device), torch_version=str(torch.__version__),
            torch_cuda_version=torch.version.cuda, cuda_available=torch.cuda.is_available(),
            gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            input_id=input_signature(data_root, args.dataset),
        )
        set_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.set_device(device)
        data = load_dataset(data_root, args.dataset)
        modalities = [name for name in CANONICAL_MODALITY_ORDER if name in data["adatas"]]
        n_clusters = data["n_clusters"]
        if n_clusters is None:
            raise ValueError("The main experiments require a fixed number of clusters.")
        config["runtime"] = {"modalities": modalities}
        config["training"]["seed"] = args.seed
        (out / "config.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

        prepared = prepare_benchmark_data(data, config)
        spatial = np.asarray(prepared.adata.obsm["spatial"], dtype=np.float32)
        bundle = build_multiscale_bundle(
            prepared.features, spatial, config["graph"]["scales"],
            bool(config["graph"].get("symmetric", False)), device,
        )
        embedding, train_info = train_corcd(
            features=prepared.features, graph_bundle=bundle, config=config,
            spatial=spatial, device=device,
        )
        embedding = embedding.detach().cpu().numpy()
        if embedding.ndim != 2 or embedding.shape[0] != len(data["obs_names"]) or not np.isfinite(embedding).all():
            raise ValueError(f"Invalid joint embedding: {embedding.shape}")
        prepared.adata.obsm["CORCD"] = embedding
        clustering(
            prepared.adata, key="CORCD", add_key="predicted_label", n_clusters=int(n_clusters),
            method="mclust", use_pca=bool(config["clustering"]["use_pca"]),
            n_comps=int(config["clustering"]["n_comps"]), random_seed=args.seed,
        )
        labels = prepared.adata.obs["predicted_label"]
        if labels.isna().any():
            raise ValueError("Clustering returned missing labels.")
        pd.DataFrame({"barcode": np.asarray(data["obs_names"], dtype=str),
                      "predicted_label": labels.astype(str).to_numpy()}).to_csv(out / "predictions.csv", index=False)
        np.save(out / "joint_embedding.npy", embedding)
        info.update(
            status="success",
            modalities=modalities, n_spots=len(data["obs_names"]), n_clusters_used=int(n_clusters),
            clustering_method="mclust", scales=sorted(config["graph"]["scales"]),
            graph_fusion=config["model"]["graph_fusion"], training=train_info,
        )
        write_info(out, info)
        print(f"[DONE] {args.dataset} seed={args.seed} -> {out}")
    except Exception as exc:
        info.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        write_info(out, info)
        raise


if __name__ == "__main__":
    main()
