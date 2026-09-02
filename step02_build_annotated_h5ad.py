"""Build the annotated AnnData object by joining the per-spot cell-type
mapping onto the clustered dataset produced in Step 1.

Cell-type identities were assigned per spot by scoring expression against
a curated panel of literature-validated marker genes (Supplementary
Table), with each spot assigned the identity of its best-supported marker
panel. The resulting per-spot assignments are provided in
data/cell_type_mapping.csv and are joined here onto the clustered object
by row position.

Inputs:
  processed_data/merged_all_treatments.h5ad -- clustered, no cell type
  data/cell_type_mapping.csv                -- cell_id -> cell_type

Output:
  processed_data/merged_annotated.h5ad -- annotated AnnData object used
  by all downstream analysis scripts.
"""
import os
import argparse
import scanpy as sc
import pandas as pd


def build(clustered_path, mapping_csv, out_path):
    print(f"Loading clustered h5ad: {clustered_path}")
    adata = sc.read_h5ad(clustered_path)
    print(f"  {adata.n_obs:,} cells")

    print(f"Loading cell type mapping: {mapping_csv}")
    mapping = pd.read_csv(mapping_csv)
    required = {"row_index", "cell_type"}
    if not required.issubset(mapping.columns):
        raise ValueError(
            f"{mapping_csv} must have {sorted(required)} columns; "
            f"found {list(mapping.columns)}")
    # Joined by row_index (0-based row position), not cell_id: spot
    # identifiers repeat across treatment samples, so row position is the
    # reliable join key.
    if mapping["row_index"].duplicated().any():
        dups = mapping.loc[mapping["row_index"].duplicated(), "row_index"].tolist()
        raise ValueError(f"Duplicate row_index values in {mapping_csv}: {dups[:5]}")

    idx_here = set(range(adata.n_obs))
    idx_mapped = set(mapping["row_index"])
    missing = idx_here - idx_mapped
    if missing:
        raise ValueError(
            f"{len(missing):,} row(s) in the h5ad have no entry in "
            f"{mapping_csv} (e.g. row_index {sorted(missing)[:5]})")
    unused = idx_mapped - idx_here
    if unused:
        print(f"  NOTE: {len(unused):,} mapping row(s) (row_index beyond "
              f"the h5ad's range) ignored")

    lut = mapping.set_index("row_index")["cell_type"]
    adata.obs["cell_type_auto"] = pd.Categorical(
        lut.reindex(range(adata.n_obs)).values)

    print("\nFinal cell type distribution:")
    counts = adata.obs["cell_type_auto"].value_counts()
    for ct, n in counts.items():
        print(f"  {ct:<35}: {n:>8,} ({100 * n / adata.n_obs:>5.1f}%)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    adata.write_h5ad(out_path)
    print(f"\nSaved: {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default=os.path.expanduser(
        "~/stereo-seq/stereoseq-analysis-bin50"))
    p.add_argument("--clustered_path", default=None,
                   help="Default: <input_dir>/downstream_analysis/processed_data/merged_all_treatments.h5ad")
    p.add_argument("--mapping_csv", default=None,
                   help="Default: <input_dir>/downstream_analysis/processed_data/cell_type_mapping.csv")
    p.add_argument("--output_path", default=None,
                   help="Default: <input_dir>/downstream_analysis/processed_data/merged_annotated.h5ad")
    args = p.parse_args()

    proc = os.path.join(args.input_dir, "downstream_analysis", "processed_data")
    data_dir = os.path.join(args.input_dir, "data")

    def resolve(explicit, filename):
        """Prefer data/<filename> (the repo's shipped copy); fall back to
        processed_data/<filename> (e.g. from an earlier local run)."""
        if explicit:
            return explicit
        candidates = [os.path.join(data_dir, filename), os.path.join(proc, filename)]
        for c in candidates:
            if os.path.exists(c):
                return c
        raise FileNotFoundError(
            f"{filename} not found in any of: {candidates}")

    clustered_path = resolve(args.clustered_path, "merged_all_treatments.h5ad")
    mapping_csv = resolve(args.mapping_csv, "cell_type_mapping.csv")
    output_path = args.output_path or os.path.join(proc, "merged_annotated.h5ad")

    build(clustered_path, mapping_csv, output_path)


if __name__ == "__main__":
    main()
