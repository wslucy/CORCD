# CORCD

Cross-omics consensus-guided contrastive learning and multi-scale relation distillation for spatial domain identification.

The main experiment uses reconstruction, consensus-guided triplet contrast and multi-scale relation distillation. Consensus ranks reciprocal feature neighbors; the triplet loss is unweighted. Six datasets use seeds 2022–2026.

## Run

Use a compatible Python environment with PyTorch, PyTorch Geometric, scanpy, SciPy, pandas, PyYAML, rpy2 and R/mclust already installed.

```bash
# All datasets and five seeds; training, evaluation and mean/std summaries
python run_all.py --data-root /path/to/datasets --device cuda:0

# One dataset, five seeds
python run_all.py --datasets misar_e15 --data-root /path/to/datasets --device cuda:0

# Preview without reading data or starting training
python run_all.py --data-root /path/to/datasets --dry-run
```

Use `--results-root` for another output directory, and `--force` to recompute completed runs. Runs are processed sequentially on the selected device.

A single training run and its evaluation are also available:

```bash
python run.py --dataset hln_a1 --seed 2022 --data-root /path/to/datasets --device cuda:0 --output-dir /path/to/run
python evaluate.py --dataset hln_a1 --seed 2022 --data-root /path/to/datasets --result-dir /path/to/run
```

## Inputs and outputs

| Dataset ID | Directory | Modalities |
|---|---|---|
| hln_a1 | 10x_human_lymph_node_A1 | RNA, ADT |
| hln_d1 | 10x_human_lymph_node_D1 | RNA, ADT |
| misar_e11 | Mouse_Brain_E11_S1 | RNA, ATAC |
| misar_e13 | Mouse_Brain_E13_S1 | RNA, ATAC |
| misar_e15 | Mouse_Brain_E15_S1 | RNA, ATAC |
| sim_sg | Sim_SG | RNA, ADT, ATAC |

Each directory contains `adata_RNA.h5ad`, the matching `adata_ADT.h5ad` / `adata_ATAC.h5ad`, and `ground_truth.csv`. AnnData requires unique spot barcodes in `obs_names` and coordinates in `obsm['spatial']`. The CSV columns are `barcode,ground_truth,ground_truth_id,is_labeled`; unlabeled spots use ID `-1`. Modalities are aligned by barcode. Labeled-class counts set the clustering K; spot labels are used for evaluation, not optimization or stopping.

Outputs include the final embedding, predicted labels, resolved configuration, run identity and metrics under `results/<dataset>/seed_<seed>/`. Five-seed mean and sample-standard-deviation tables are under `results/summary/`. The five metrics are ARI, NMI, FMI, Homogeneity and Completeness, all multiplied by 100.

This folder contains the main-experiment implementation and fixed settings.

Re-evaluate saved predictions without training:

```bash
python evaluate.py --data-root /path/to/datasets --results-root /path/to/results
```
