"""Load the six paired spatial multi-omics datasets used in the main experiments.

Modalities are aligned by barcode and share the RNA spatial coordinates. Labels
and the labeled-spot mask are returned as evaluation metadata; model fitting
must not use them.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd

from .dataset_registry import BENCHMARK_SEEDS, DATASET_CONFIG


def get_dataset_config(dataset_id):
    if dataset_id not in DATASET_CONFIG:
        raise KeyError(
            f"Unknown dataset_id={dataset_id!r}. "
            f"Available: {tuple(DATASET_CONFIG)}"
        )
    return DATASET_CONFIG[dataset_id]


def _parse_bool(series):
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin({"true", "1", "yes"})
    )


def load_ground_truth(dataset_dir, obs_names):
    """Load and barcode-align ``ground_truth.csv``."""
    gt_path = Path(dataset_dir) / "ground_truth.csv"
    if not gt_path.exists():
        raise FileNotFoundError(gt_path)

    gt = pd.read_csv(gt_path, dtype={"barcode": str})
    required = {"barcode", "ground_truth", "ground_truth_id", "is_labeled"}
    missing = required - set(gt.columns)
    if missing:
        raise ValueError(f"{gt_path}: missing columns {sorted(missing)}")
    if gt["barcode"].duplicated().any():
        raise ValueError(f"{gt_path}: duplicated barcode")
    if gt["barcode"].isna().any():
        raise ValueError(f"{gt_path}: barcode contains missing values")

    obs_names = pd.Index(obs_names, dtype=str)
    if not obs_names.is_unique:
        raise ValueError("Dataset barcodes must be unique")
    gt = gt.set_index("barcode").reindex(obs_names)
    if gt["is_labeled"].isna().any():
        count = int(gt["is_labeled"].isna().sum())
        raise ValueError(
            f"{gt_path}: {count} benchmark spots are absent from ground_truth.csv"
        )

    label_mask = _parse_bool(gt["is_labeled"]).to_numpy()
    if gt.loc[label_mask, "ground_truth"].isna().any():
        raise ValueError(f"{gt_path}: labeled spots contain missing labels")

    unlabeled_ids = pd.to_numeric(
        gt.loc[~label_mask, "ground_truth_id"], errors="coerce"
    )
    if len(unlabeled_ids) and not (unlabeled_ids == -1).all():
        raise ValueError(f"{gt_path}: unlabeled ground_truth_id must be -1")

    return {
        "ground_truth": gt,
        "label_mask": label_mask,
        "y_true": gt.loc[label_mask, "ground_truth"].astype(str).to_numpy(),
        "y_true_id": pd.to_numeric(
            gt.loc[label_mask, "ground_truth_id"], errors="raise"
        ).astype(int).to_numpy(),
        "n_clusters": int(gt.loc[label_mask, "ground_truth"].nunique()),
    }


def _read_metadata(path):
    import scanpy as sc

    adata = sc.read_h5ad(path, backed="r")
    try:
        obs_names = pd.Index(adata.obs_names.astype(str))
        if not obs_names.is_unique:
            raise ValueError(f"{path}: duplicated obs_names")
        if "spatial" not in adata.obsm:
            raise KeyError(f"{path}: obsm['spatial'] is missing")
        spatial = np.asarray(adata.obsm["spatial"]).copy()
    finally:
        try:
            adata.file.close()
        except Exception:
            pass
    return obs_names, spatial


def load_evaluation_data(data_root, dataset_id):
    """Load only barcodes, spatial coordinates and GT for evaluation."""
    data_root = Path(data_root).expanduser().resolve()
    config = get_dataset_config(dataset_id)
    dataset_dir = data_root / config["path"]
    if not dataset_dir.is_dir():
        raise FileNotFoundError(dataset_dir)

    reference_obs = None
    reference_spatial = None

    for modality, filename in config["modalities"].items():
        path = dataset_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)
        obs_names, spatial = _read_metadata(path)

        if reference_obs is None:
            reference_obs = obs_names
            reference_spatial = spatial
            continue

        if set(obs_names) != set(reference_obs):
            raise ValueError(f"{dataset_id}/{modality}: spot set mismatch")

        position = pd.Series(np.arange(len(obs_names)), index=obs_names)
        order = position.loc[reference_obs].to_numpy()
        aligned_spatial = spatial[order]
        if not np.allclose(
            aligned_spatial,
            reference_spatial,
            rtol=1e-5,
            atol=1e-6,
            equal_nan=True,
        ):
            warnings.warn(
                f"{dataset_id}/{modality}: spatial coordinates differ from the "
                "canonical coordinates. CORCD will use the first "
                "configured modality (normally RNA) as the shared spatial "
                "coordinate system.",
                RuntimeWarning,
            )

    result = {
        "dataset_id": dataset_id,
        "dataset_dir": dataset_dir,
        "config": config,
        "obs_names": reference_obs,
        "spatial": reference_spatial,
        "spatial_source": next(iter(config["modalities"])),
        "supervised": bool(config["supervised"]),
        "ground_truth": None,
        "label_mask": None,
        "y_true": None,
        "y_true_id": None,
        "n_clusters": None,
    }

    if config["supervised"]:
        result.update(load_ground_truth(dataset_dir, reference_obs))
    return result


def load_dataset(data_root, dataset_id):
    """Load aligned AnnData modalities and metadata without normalization.

    Ground-truth labels are evaluation metadata and must not enter model fitting.
    """
    import scanpy as sc

    data_root = Path(data_root).expanduser().resolve()
    config = get_dataset_config(dataset_id)
    dataset_dir = data_root / config["path"]
    if not dataset_dir.is_dir():
        raise FileNotFoundError(dataset_dir)

    adatas = {}
    reference_obs = None
    reference_spatial = None

    for modality, filename in config["modalities"].items():
        path = dataset_dir / filename
        if not path.exists():
            raise FileNotFoundError(path)

        adata = sc.read_h5ad(path)
        if not adata.var_names.is_unique:
            adata.var_names_make_unique()
        if not adata.obs_names.is_unique:
            raise ValueError(f"{dataset_id}/{modality}: duplicated obs_names")
        if "spatial" not in adata.obsm:
            raise KeyError(f"{dataset_id}/{modality}: obsm['spatial'] is missing")

        obs_names = pd.Index(adata.obs_names.astype(str))
        if reference_obs is None:
            reference_obs = obs_names
            reference_spatial = np.asarray(adata.obsm["spatial"])
        else:
            if set(obs_names) != set(reference_obs):
                raise ValueError(f"{dataset_id}/{modality}: spot set mismatch")
            if not obs_names.equals(reference_obs):
                adata = adata[reference_obs].copy()
            modality_spatial = np.asarray(adata.obsm["spatial"])
            if not np.allclose(
                modality_spatial,
                reference_spatial,
                rtol=1e-5,
                atol=1e-6,
                equal_nan=True,
            ):
                warnings.warn(
                    f"{dataset_id}/{modality}: spatial coordinates differ from "
                    "the canonical coordinates. Keeping the original values in "
                    "adata.obsm['spatial_original'] and using the first configured "
                    "modality (normally RNA) as the shared spatial coordinates.",
                    RuntimeWarning,
                )
                if "spatial_original" not in adata.obsm:
                    adata.obsm["spatial_original"] = modality_spatial.copy()
                adata.obsm["spatial"] = reference_spatial.copy()

        adatas[modality] = adata

    result = {
        "dataset_id": dataset_id,
        "dataset_dir": dataset_dir,
        "config": config,
        "adatas": adatas,
        "obs_names": reference_obs,
        "spatial": reference_spatial,
        "spatial_source": next(iter(config["modalities"])),
        "supervised": bool(config["supervised"]),
        "ground_truth": None,
        "label_mask": None,
        "y_true": None,
        "y_true_id": None,
        "n_clusters": None,
    }

    if config["supervised"]:
        result.update(load_ground_truth(dataset_dir, reference_obs))
    return result
