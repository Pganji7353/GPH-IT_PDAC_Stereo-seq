#!/usr/bin/env bash
# Runs step02 through step13 in order, using data/ as the input reference
# and the current directory for all intermediate/output files. Stops on the
# first failure. Logs go to logs/<step>.log.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

echo "=== Pre-flight checks ==="
if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda not found on PATH. Install Miniconda/Anaconda first."
    exit 1
fi
if ! conda env list | awk '{print $1}' | grep -qx "stereoseq_local"; then
    echo "ERROR: conda environment 'stereoseq_local' not found."
    echo "Run ./setup_env.sh first to create it."
    exit 1
fi
for f in data/merged_all_treatments.h5ad data/cell_type_mapping.csv; do
    if [ ! -f "$f" ]; then
        echo "ERROR: required input file missing: $f"
        exit 1
    fi
done
echo "  conda env 'stereoseq_local' found, required data/ files present."

mkdir -p logs downstream_analysis/processed_data

run() {
    local name="$1"; shift
    echo ""
    echo "=== $name ==="
    local start=$(date +%s)
    if conda run -n stereoseq_local --no-capture-output python "$@" > "logs/${name}.log" 2>&1; then
        local elapsed=$(( $(date +%s) - start ))
        echo "$name done in ${elapsed}s"
    else
        echo "$name FAILED, see logs/${name}.log"
        tail -30 "logs/${name}.log"
        exit 1
    fi
}

run step02 step02_build_annotated_h5ad.py --input_dir .
# --deg_only re-runs just step01's DEG figures (Fig 6H, 6I) against Step 2's
# annotated h5ad, since the full step01 pipeline needs raw SAW data this repo
# does not ship.
run step01_deg step01_qc_integration_clustering_and_DEG.py --base_dir . --deg_only
run step03 step03_subcluster_ecotype_treatment_trends.py --input_dir . --output_dir .
run step04 step04_ecotype_panels.py --input_dir .
run step05 step05_cell_distances.py --input_dir .
run step06 step06_umap_lineage_plots.py --input_dir .
run step07 step07_cell_subtype_plots.py \
    --h5ad_path downstream_analysis/processed_data/merged_annotated.h5ad \
    --out_dir downstream_analysis/figures/post6_subtype_plots
run step08 step08_annotation_summary_plots.py \
    --h5ad_path downstream_analysis/processed_data/merged_annotated.h5ad \
    --out_dir downstream_analysis/figures/annotation_summary
run step10 step10_pdac_spatial_analysis.py --input_dir .
run step11 step11_sankey_plots.py \
    --ecotype_csv downstream_analysis/processed_data/unified_ecotype_assignments.csv \
    --out_dir downstream_analysis/figures/panels_ecotype
run step12 step12_interaction_pathway_analysis.py --input_dir . --output_dir .
# step09 must run AFTER step12: it needs lr_interactions_per_treatment.csv,
# which step12 (LIANA) produces. Running step09 first silently skips the
# CellChat circos figures instead of failing.
run step09 step09_cellchat_interaction_panels.py --input_dir .
run step13 step13_cnv_spatial_niche_analysis.py --input_dir . --output_dir .

echo ""
echo "=== All steps complete. Results in downstream_analysis/figures/ ==="
