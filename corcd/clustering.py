"""Fixed mclust postprocessing for CORCD joint embeddings."""

from __future__ import annotations

import numpy as np

from .preprocessing import pca


def mclust_R(adata, num_cluster, modelNames="EEE", used_obsm="emb_pca", random_seed=2020):
    """
    Perform clustering using R package `mclust`.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object containing representation in `.obsm`.
    num_cluster : int
        Number of clusters.
    modelNames : str, default="EEE"
        Model type in `mclust`.
    used_obsm : str, default="emb_pca"
        Key in `adata.obsm` to use for clustering.
    random_seed : int, default=2020
        Random seed for reproducibility.

    Returns
    -------
    adata : anndata.AnnData
        Updated AnnData with `adata.obs['mclust']`.
    """
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri

    robjects.r.library("mclust")
    rpy2.robjects.numpy2ri.activate()

    r_random_seed = robjects.r["set.seed"]
    r_random_seed(random_seed)
    rmclust = robjects.r["Mclust"]

    res = rmclust(rpy2.robjects.numpy2ri.numpy2rpy(adata.obsm[used_obsm]), num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata


def clustering(
    adata,
    n_clusters=7,
    key="emb",
    add_key="CORCD",
    method="mclust",
    use_pca=False,
    n_comps=20,
    random_seed=0,
):
    """Cluster an embedding with mclust's EEE covariance model."""
    if method != "mclust":
        raise ValueError("CORCD main experiments use mclust clustering")
    if use_pca:
        adata.obsm[key + "_pca"] = pca(adata, use_reps=key, n_comps=n_comps)
    adata = mclust_R(
        adata,
        used_obsm=(key + "_pca" if use_pca else key),
        num_cluster=n_clusters,
        random_seed=int(random_seed),
    )
    adata.obs[add_key] = adata.obs["mclust"]
