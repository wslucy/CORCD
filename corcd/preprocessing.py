"""Modality preprocessing for the CORCD main experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import scanpy as sc
import torch
from muon import atac as ac
from muon import prot as pt


@dataclass
class PreparedData:
    adata: object
    modality_adatas: List[object]
    features: List[torch.Tensor]
    triplet_samples_list: list
    gt: Optional[List[str]]
    n_clusters: Optional[int]
    default_spatial_k: int


def pca(adata, use_reps=None, n_comps=10):
    """
    Perform dimensionality reduction using PCA.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object.
    use_reps : str, optional
        Key in `adata.obsm` to use as input features.
        If None, use `adata.X`.
    n_comps : int, default=10
        Number of principal components.

    Returns
    -------
    np.ndarray
        PCA-reduced features of shape [n_samples, n_comps].
    """
    from sklearn.decomposition import PCA
    from scipy.sparse import csc_matrix, csr_matrix

    pca = PCA(n_components=n_comps)
    if use_reps is not None:
        feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else:
        if isinstance(adata.X, (csc_matrix, csr_matrix)):
            feat_pca = pca.fit_transform(adata.X.toarray())
        else:
            feat_pca = pca.fit_transform(adata.X)

    return feat_pca


CANONICAL_MODALITY_ORDER = ("RNA", "ADT", "ATAC")


def _safe_n_comps(adata, requested: int) -> int:
    return max(2, min(int(requested), int(adata.n_obs) - 1, int(adata.n_vars)))


def _prepare_rna(adata, n_hvg: int, n_comps: int):
    adata = adata.copy()
    adata.var_names_make_unique()
    sc.pp.filter_genes(adata, min_cells=min(10, max(1, adata.n_obs // 20)))
    n_hvg = min(int(n_hvg), int(adata.n_vars))
    try:
        sc.pp.highly_variable_genes(adata, flavor="seurat_v3", n_top_genes=n_hvg)
    except Exception:
        sc.pp.highly_variable_genes(adata, flavor="seurat", n_top_genes=n_hvg)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata)
    hvg = adata.var.get("highly_variable")
    view = adata[:, np.asarray(hvg, dtype=bool)] if hvg is not None and np.any(hvg) else adata
    view.obsm["feat"] = pca(view, n_comps=_safe_n_comps(view, n_comps))
    adata.obsm["feat"] = view.obsm["feat"]
    return adata


def _prepare_adt(adata, n_comps: int):
    adata = adata.copy()
    adata.var_names_make_unique()
    pt.pp.clr(adata)
    sc.pp.scale(adata)
    adata.obsm["feat"] = pca(adata, n_comps=_safe_n_comps(adata, n_comps))
    return adata


def _prepare_atac(adata, n_comps: int):
    adata = adata.copy()
    adata.var_names_make_unique()
    ac.pp.tfidf(adata, scale_factor=1e4)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    adata.obsm["feat"] = pca(adata, n_comps=_safe_n_comps(adata, n_comps))
    return adata


def prepare_benchmark_data(data: Dict, config: Dict) -> PreparedData:
    """Convert barcode-aligned dataset modalities into CORCD inputs.

    Ground-truth columns are intentionally ignored. Only ``n_clusters`` may be
    used later by a clustering backend when the benchmark marks a dataset as
    supervised.
    """

    prep_cfg = config.get("preprocessing", {})
    adatas = data["adatas"]
    ordered_names = [name for name in CANONICAL_MODALITY_ORDER if name in adatas]
    if len(ordered_names) < 2:
        raise ValueError(f"The spatial multi-omics model requires at least two modalities, got {ordered_names}")

    prepared: List[object] = []
    for name in ordered_names:
        adata = adatas[name]
        if name == "RNA":
            out = _prepare_rna(
                adata,
                n_hvg=int(prep_cfg.get("rna_hvg", 3000)),
                n_comps=int(prep_cfg.get("rna_pca", 30)),
            )
        elif name == "ADT":
            out = _prepare_adt(adata, n_comps=int(prep_cfg.get("adt_pca", 30)))
        elif name == "ATAC":
            out = _prepare_atac(adata, n_comps=int(prep_cfg.get("atac_pca", 60)))
        else:  # pragma: no cover
            raise ValueError(f"Unsupported modality: {name}")
        prepared.append(out)

    reference_names = np.asarray(data["obs_names"], dtype=str)
    for idx, adata in enumerate(prepared):
        if not np.array_equal(np.asarray(adata.obs_names, dtype=str), reference_names):
            prepared[idx] = adata[reference_names].copy()

    reference = prepared[0].copy()
    reference.obsm["spatial"] = np.asarray(data["spatial"], dtype=np.float32)
    features = [torch.as_tensor(x.obsm["feat"], dtype=torch.float32) for x in prepared]

    triplets = [None for _ in prepared]

    default_k = int(config.get("graph", {}).get("spatial_k") or 6)
    result = PreparedData(
        adata=reference,
        modality_adatas=prepared,
        features=features,
        triplet_samples_list=triplets,
        gt=None,
        n_clusters=data.get("n_clusters"),
        default_spatial_k=default_k,
    )
    result.modality_names = ordered_names
    result.obs_names = reference_names.tolist()
    return result
