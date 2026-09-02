#!/usr/bin/env bash
# Creates the "stereoseq_local" conda environment and installs every package
# this pipeline needs, then verifies the environment is ready to run
# generate_results.sh.
#
# squidpy pulls in a newer numpy than the one pinned in requirements.txt and
# upgrades it in place. That numpy 2.x build is ABI-incompatible with the
# numba-compiled code paths scanpy and BANKSY rely on (observed as a hard
# segfault deep into step13, well past normal Python error handling), so
# numpy is pinned back down to 1.26.4 as the last install step. banksy_py
# pins an old dask release that conflicts with squidpy's real dependency,
# so it is installed with --no-deps. Recent setuptools also dropped the
# pkg_resources module that spatialdata (a squidpy dependency) still
# imports, so setuptools is pinned below 81 to keep it available.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

ENV_NAME="stereoseq_local"
CONDA_BASE="$(conda info --base)"
PY="$CONDA_BASE/envs/$ENV_NAME/bin/python"
PIP="$CONDA_BASE/envs/$ENV_NAME/bin/pip"

echo "=== Creating conda environment: $ENV_NAME ==="
conda create -y -n "$ENV_NAME" python=3.11

echo ""
echo "=== Installing core scientific + single-cell packages ==="
"$PIP" install \
    numpy==1.26.4 pandas==2.3.3 scipy==1.16.3 scikit-learn==1.8.0 \
    statsmodels==0.14.6 networkx==3.6.1 matplotlib==3.8.4 seaborn==0.13.2 \
    statannotations==0.7.2 scanpy==1.11.5 anndata==0.12.6 harmonypy==0.2.0 \
    gseapy==1.1.12 scrublet==0.2.3

echo ""
echo "=== Installing squidpy (upgrades numpy; expected) ==="
"$PIP" install squidpy==1.7.0

echo ""
echo "=== Installing infercnvpy and liana ==="
"$PIP" install infercnvpy==0.6.1 liana==1.6.1

echo ""
echo "=== Pinning setuptools <81 (spatialdata still needs pkg_resources) ==="
"$PIP" install "setuptools<81"

echo ""
echo "=== Installing banksy_py (no-deps: its own dask pin conflicts with squidpy's) ==="
"$PIP" install banksy_py==0.0.1 --no-deps

echo ""
echo "=== Repinning numpy==1.26.4 (squidpy silently upgraded it; the newer build segfaults numba-compiled code) ==="
"$PIP" install numpy==1.26.4

echo ""
echo "=== Verifying environment ==="
"$PY" - <<'PYEOF'
import sys
mods = ["scanpy", "anndata", "squidpy", "harmonypy", "infercnvpy", "liana",
        "gseapy", "banksy", "numpy", "pandas", "scipy", "matplotlib",
        "seaborn", "sklearn", "statsmodels", "networkx", "statannotations",
        "scrublet"]
ok = True
for m in mods:
    try:
        mod = __import__(m)
        v = getattr(mod, "__version__", "?")
        print(f"  {m:<16} {v}")
    except Exception as e:
        print(f"  {m:<16} FAILED: {e}")
        ok = False

if not ok:
    print("\nEnvironment is NOT ready: one or more packages failed to import.")
    sys.exit(1)

print("\nEnvironment is ready. Activate it with:")
print(f"  conda activate stereoseq_local")
print("Then run: ./generate_results.sh")
PYEOF
