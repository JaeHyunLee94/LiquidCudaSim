#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# bench.slurm — build & run the MPM benchmark on an HPC node.
#
# Submit with:
#   sbatch scripts/bench.slurm
#
# Outputs:
#   results/bench-<jobid>.csv     — one CSV per submission, all configs appended
#   slurm-<jobid>.out             — stdout/stderr of the run
#
# Tunable env vars (override at submit time, e.g. `STEPS=500 sbatch …`):
#   SCENE      cube|bunny       default cube
#   STEPS      timed steps      default 200
#   WARMUP     untimed steps    default 20
#   DT         timestep         default 1e-4
#   GRID       grid resolution  default 64
#   OMP_LIST   space-separated  thread counts to sweep, default "1 2 4 8 16"
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=mpm-bench
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.out
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16          # OpenMP runs use up to this many
#SBATCH --gres=gpu:1                # one GPU for the CUDA build
#SBATCH --mem=16G

set -euo pipefail

# ─── Knobs (overridable at submit time) ──────────────────────────────────────
SCENE="${SCENE:-cube}"
STEPS="${STEPS:-200}"
WARMUP="${WARMUP:-20}"
DT="${DT:-1e-4}"
GRID="${GRID:-64}"
OMP_LIST="${OMP_LIST:-1 2 4 8 16}"

# ─── Paths ───────────────────────────────────────────────────────────────────
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${PROJECT_ROOT}/results"
CSV="${RESULTS_DIR}/bench-${SLURM_JOB_ID:-local}.csv"
mkdir -p "${RESULTS_DIR}"

# ─── Modules (edit for your cluster) ─────────────────────────────────────────
# These names are the most common defaults on US/EU HPC sites. If your cluster
# uses different module names, override here. Module load is best-effort:
# missing modules don't abort the job (the underlying tool may already be on PATH).
module purge 2>/dev/null || true
module load cmake          2>/dev/null || true
module load gcc/11         2>/dev/null || true
module load cuda/12.2      2>/dev/null || true   # any 11.x/12.x works

# ─── Banner ──────────────────────────────────────────────────────────────────
echo "==============================================================="
echo "MPM benchmark — job ${SLURM_JOB_ID:-local}  on $(hostname)"
echo "  scene=${SCENE} steps=${STEPS} warmup=${WARMUP} dt=${DT} grid=${GRID}"
echo "  omp sweep: ${OMP_LIST}"
echo "  CSV: ${CSV}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
    || echo "  (nvidia-smi not available — GPU run will be skipped if no compiler)"
echo "==============================================================="

# ─── 1. Build CUDA flavour (headless) ────────────────────────────────────────
build_cuda_dir="${PROJECT_ROOT}/build-bench-cuda"
echo "[build] configuring CUDA build at ${build_cuda_dir}"
cmake -S "${PROJECT_ROOT}" -B "${build_cuda_dir}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DMPM_ENABLE_CUDA=ON \
      -DMPM_ENABLE_DISPLAY=OFF
cmake --build "${build_cuda_dir}" --target benchmark -j

# ─── 2. Build OpenMP-only flavour (headless) ─────────────────────────────────
build_omp_dir="${PROJECT_ROOT}/build-bench-omp"
echo "[build] configuring OpenMP build at ${build_omp_dir}"
cmake -S "${PROJECT_ROOT}" -B "${build_omp_dir}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DMPM_ENABLE_CUDA=OFF \
      -DMPM_ENABLE_DISPLAY=OFF
cmake --build "${build_omp_dir}" --target benchmark -j

# ─── 3. Helper: run the benchmark from the scene/ directory so the relative
#       asset paths (../../assets/*.bgeo) resolve correctly. ─────────────────
run_bench() {
    local exe="$1"; shift
    (
        cd "$(dirname "${exe}")"
        "./$(basename "${exe}")" "$@"
    )
}

CUDA_EXE="${build_cuda_dir}/scene/benchmark"
OMP_EXE="${build_omp_dir}/scene/benchmark"

COMMON_ARGS=(--scene "${SCENE}" --steps "${STEPS}" --warmup "${WARMUP}" \
             --dt "${DT}" --grid "${GRID}" --csv "${CSV}")

# ─── 4. CUDA run (single launch — GPU only has one parallelism dimension) ────
if [[ -x "${CUDA_EXE}" ]]; then
    echo
    echo "------------- CUDA run -------------"
    run_bench "${CUDA_EXE}" "${COMMON_ARGS[@]}" --tag "cuda"
else
    echo "[skip] CUDA binary not built — likely no CUDA toolkit on this node"
fi

# ─── 5. OpenMP runs — sweep over thread counts to get scaling data ──────────
if [[ -x "${OMP_EXE}" ]]; then
    for nt in ${OMP_LIST}; do
        echo
        echo "------------- OpenMP run, threads=${nt} -------------"
        OMP_NUM_THREADS="${nt}" \
        OMP_PROC_BIND=close OMP_PLACES=cores \
            run_bench "${OMP_EXE}" "${COMMON_ARGS[@]}" --tag "omp-${nt}"
    done
else
    echo "[skip] OpenMP binary missing"
fi

echo
echo "==============================================================="
echo "Done. Aggregated results:"
echo "  ${CSV}"
echo "==============================================================="
column -t -s, "${CSV}" | sed 's/^/  /'
