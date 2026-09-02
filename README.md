# Spatial and TCR Profiling of GPH Plus Immune Checkpoint Therapy in Pancreatic Ductal Adenocarcinoma

Stereo-seq analysis pipeline for this study. The companion single-cell TCR repertoire
analysis lives in a separate repository:
[madhubioinformatics/scTCR_PDAC](https://github.com/madhubioinformatics/scTCR_PDAC).

## Overview

PDAC has a tumor microenvironment (TME) that shuts down immune activity, which is a big part
of why it's so hard to treat. Earlier work from our group showed that gemcitabine,
hydroxychloroquine, and paricalcitol (GPH) can modulate that immune microenvironment.

This study asks what happens when GPH is combined with immune checkpoint therapy (IT;
anti-PD-1 + anti-CTLA-4). We used Stereo-seq alongside single-cell TCR (scTCR) sequencing to
track the spatial, cellular, and clonal remodeling of the PDAC TME after treatment.

## Study design

Orthotopic PDAC mouse models, four treatment groups:

- **Sham** -- control
- **IT** -- anti-PD-1 + anti-CTLA-4 immune checkpoint therapy
- **GPH** -- gemcitabine + hydroxychloroquine + paricalcitol
- **GPH+IT** -- both combined

Stereo-seq tracked spatial changes in the TME; scTCR sequencing (companion repo) tracked
T-cell clonality, expansion, phenotype, and treatment-associated immune response.

## Key findings

GPH+IT produced the strongest antitumor response of any group, with more tumor growth
inhibition than either treatment alone. In the survival study, GPH+IT was the only group with
complete tumor remission.

T cells pulled from treated mice kept their antitumor activity and controlled orthotopic PDAC
tumors in both immunocompetent and RagKO mice.

scTCR analysis found T-cell clones -- especially effector CD4+ clones -- enriched at the
tumor-immune interface, and these clones expanded substantially after T-cell infusion. Instead
of the terminal exhaustion you might expect, the expanded clones showed signs of effector
activation, plus more memory CD8+ and proliferative T cells.

Stereo-seq showed extensive spatial remodeling of the TME after GPH+IT:

- M1 macrophages went from essentially undetectable in sham to ~7.5% of all profiled cells
  (~83% of macrophages) after GPH+IT.
- CD8+ and cytotoxic T cells moved closer to the tumor, consistent with better immune access.
- Tumor hypoxia scores dropped.
- Spatial ecotype analysis showed GPH+IT reorganized tumors into more open,
  immune-accessible layouts not seen in the other groups.

## Biological significance

GPH+IT drives coordinated clonal, cellular, and spatial remodeling of the PDAC TME: more
immune infiltration, macrophages shifting toward M1, less hypoxia, and antitumor T-cell clones
expanding and activating.

Combining Stereo-seq with scTCR profiling connects T-cell clonal dynamics to where those cells
sit spatially and what functional state they're in -- which gives some mechanistic insight
into how GPH plus checkpoint blockade works, and points toward spatial/immune biomarkers of
treatment response. It also makes a case for pairing this approach with TCR-based
immunotherapies and adoptive T-cell transfer.

## Repository contents

This repository (Stereo-seq side) contains the computational workflows and analysis code for:

- Stereo-seq preprocessing and quality control
- Spatial cell-type annotation
- Tumor-immune spatial interaction analysis
- Spatial ecotype analysis
- Tumor hypoxia analysis
- Manuscript figure generation and supporting analyses

The scTCR repertoire analysis (clonotype expansion, phenotype, functional-state
characterization) is in the companion repo linked above.

## Citation

Citation info will be added once the manuscript is published.

## Setup

1. Pick one working directory and download everything into it:
   - Raw and processed Stereo-seq data from GEO accession
     [GSE345064](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE345064)
     (4 samples: Sham, GPH, IT, GPH+IT, download each GSM record's supplementary files).
     If the accession is still private, request reviewer access via the GEO record page.
   - `data/merged_all_treatments.h5ad` and `data/merged_annotated.h5ad` (the clustered and
     cell-type-annotated merged AnnData objects across all four treatments -- too large for
     this repository) from Zenodo:
     [doi.org/10.5281/zenodo.22246345](https://doi.org/10.5281/zenodo.22246345). Only
     `merged_all_treatments.h5ad` is required to run the pipeline (step02 builds
     `merged_annotated.h5ad` from it); the annotated file is provided for convenience if you
     want to skip step02.
   - All other files from this repository, including `data/cell_type_mapping.csv` and
     `data/SupplTable_marker_gene_panels.csv`.

   Scripts, data, and outputs all live in that one directory.

2. Set up the environment:
   ```bash
   ./setup_env.sh
   ```
   This creates a conda environment named `stereoseq_local`, installs every package in
   `requirements.txt` (plus a couple of pins needed to resolve real dependency conflicts
   between squidpy and banksy_py, explained in comments in the script), and finishes by
   verifying every package imports correctly before declaring the environment ready.

## How to run

Cell-type assignment (step01's clustering output plus `data/cell_type_mapping.csv`) has
already been done for you; `data/merged_all_treatments.h5ad` in this repo is that clustering
output. The full step01 pipeline (QC, Harmony integration, Leiden clustering) needs the raw
SAW output, which this repo does not ship, so `generate_results.sh` runs steps 2 through 13
plus a `--deg_only` fast path of step01 that regenerates just its DEG figures (Fig 6H, 6I)
from step02's annotated h5ad. Two ways to run it:

**One command:**
```bash
conda activate stereoseq_local
./generate_results.sh
```
`step02_build_annotated_h5ad.py` reads `data/merged_all_treatments.h5ad` and
`data/cell_type_mapping.csv` directly (no copy or symlink) and writes
`merged_annotated.h5ad` into `downstream_analysis/processed_data/`, which every later step
reads from there. `generate_results.sh` runs every step in order, stopping immediately and
printing the error if any step fails. Per-step logs land in `logs/`.

**Or run steps individually**, `cd` into the working directory first (all commands below use
`.` for that directory):

```bash
python step02_build_annotated_h5ad.py --input_dir .
python step01_qc_integration_clustering_and_DEG.py --base_dir . --deg_only
python step03_subcluster_ecotype_treatment_trends.py --input_dir . --output_dir .
python step04_ecotype_panels.py --input_dir .
python step05_cell_distances.py --input_dir .
python step06_umap_lineage_plots.py --input_dir .
python step07_cell_subtype_plots.py --h5ad_path downstream_analysis/processed_data/merged_annotated.h5ad --out_dir downstream_analysis/figures/post6_subtype_plots
python step08_annotation_summary_plots.py --h5ad_path downstream_analysis/processed_data/merged_annotated.h5ad --out_dir downstream_analysis/figures/annotation_summary
python step10_pdac_spatial_analysis.py --input_dir .
python step11_sankey_plots.py --ecotype_csv downstream_analysis/processed_data/unified_ecotype_assignments.csv --out_dir downstream_analysis/figures/panels_ecotype
python step12_interaction_pathway_analysis.py --input_dir . --output_dir .
# step09 must run AFTER step12 (needs lr_interactions_per_treatment.csv, which
# step12/LIANA produces) -- running it earlier silently skips the CellChat
# circos figures instead of failing.
python step09_cellchat_interaction_panels.py --input_dir .
python step13_cnv_spatial_niche_analysis.py --input_dir . --output_dir .
```

(To run the full step01 pipeline from raw SAW output instead of the `--deg_only` fast path:
`python step01_qc_integration_clustering_and_DEG.py --base_dir raw_data --output_dir .`.)

| Script | Produces | Depends on |
|---|---|---|
| `step02_build_annotated_h5ad.py` | `merged_annotated.h5ad` | `merged_all_treatments.h5ad` + `data/cell_type_mapping.csv` (the definitive per-spot cell-type assignments; see Methods) |
| `step01_qc_integration_clustering_and_DEG.py --deg_only` | Fig 6H, 6I | `merged_annotated.h5ad` from step02 |
| `step03_subcluster_ecotype_treatment_trends.py` | Macrophage/T cell/CAF subclustering, spatial ecotype detection, treatment composition trends | `merged_annotated.h5ad` |
| `step04_ecotype_panels.py` | Fig 6A, 6C, 6D, 6E, 6F; Suppl. Fig 9 | `merged_annotated.h5ad`, ecotype assignments from step03 |
| `step05_cell_distances.py` | Fig 7B | `merged_annotated.h5ad`, ecotype assignments |
| `step06_umap_lineage_plots.py` | Suppl. Fig 8 | `merged_annotated.h5ad` |
| `step07_cell_subtype_plots.py` | Fig 7C, 7D, 7E; Suppl. Fig 12 | `merged_annotated.h5ad` |
| `step08_annotation_summary_plots.py` | Fig 6B | `merged_annotated.h5ad` |
| `step09_cellchat_interaction_panels.py` | Fig 7A | `merged_annotated.h5ad`, `lr_interactions_per_treatment.csv` from step12 |
| `step10_pdac_spatial_analysis.py` | Fig 7F, 7G, 7H; Suppl. Fig 13A | `merged_annotated.h5ad` |
| `step11_sankey_plots.py` | Fig 6G | `merged_annotated.h5ad`, `unified_ecotype_assignments.csv` from step04 |
| `step12_interaction_pathway_analysis.py` | Suppl. Fig 10A, 11A, 11B | `merged_annotated.h5ad` |
| `step13_cnv_spatial_niche_analysis.py` | Suppl. Fig 10B, 13B | `merged_annotated.h5ad` |

## Output layout

- **`results/`** (top level): the curated set of figures actually used in the paper, both
  `.png` and `.pdf`, named to match the manuscript's figure numbering (`fig6_a.png`,
  `suppl9_sham.pdf`, etc.). Each step script copies its own paper figures here as soon as it
  finishes, via the shared `collect_results.py` helper, so this folder fills in incrementally
  as `generate_results.sh` runs rather than only at the end.
- **`downstream_analysis/figures/`**: every figure each script produces, including
  diagnostic and exploratory panels not used in the paper (one subfolder per script).
- **`downstream_analysis/processed_data/`**: intermediate data files (`merged_all_treatments.h5ad`,
  `merged_annotated.h5ad`, and CSVs read by later steps).

## Notes

- `step04` must run before `step11` (Sankey plot needs `step04`'s ecotype assignment CSV).
- `step12` must run before `step09` (CellChat circos needs step12/LIANA's
  `lr_interactions_per_treatment.csv`; `generate_results.sh` already runs them in this order).
