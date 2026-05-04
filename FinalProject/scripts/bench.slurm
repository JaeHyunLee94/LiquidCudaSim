#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# bench.slurm — build & run the MPM benchmark on UW-Madison Euler.
#
# Submit:
#   sbatch scripts/bench.slurm
#
# Outputs:
#   results/bench-<jobid>.csv     — CSV (one row per backend/thread config)
#   bench-<jobid>.out / .err      — stdout / stderr
#
# Tunable env vars (override at submit time, e.g. `STEPS=500 sbatch …`):
#   SCENE      cube|bunny       default cube
#   STEPS      timed steps      default 200
#   WARMUP     untimed steps    default 20
#   DT         timestep         default 1e-4
#   GRID       grid resolution  default 64
#   OMP_LIST   thread counts    default "1 2 4 8 16"
# ─────────────────────────────────────────────────────────────────────────────

#SBATCH -p instruction              # class partition (priority for students)
#SBATCH -J mpm-bench
#SBATCH -o bench-%j.out
#SBATCH -e bench-%j.err
#SBATCH -t 00:30:00
#SBATCH -c 16                       # OpenMP sweep tops out around here
#SBATCH --mem=16G
# Pin a GPU type, or use `--gres=gpu:1` for any-free. Override at submit time:
#     sbatch --gres=gpu:a100:1     scripts/bench.slurm
#     sbatch --gres=gpu:h100:1     scripts/bench.slurm
#     sbatch --gres=gpu:rtxa4500:1 scripts/bench.slurm
#SBATCH --gres=gpu:1

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

# ─── Modules ─────────────────────────────────────────────────────────────────
# Match the repo759 HW pattern: load CUDA only — gcc + cmake on Euler are on
# PATH by default. Override CUDA_MODULE in env to test other versions.
CUDA_MODULE="${CUDA_MODULE:-nvidia/cuda/13.0.0}"
module load "${CUDA_MODULE}"

# Verify tools (handy when a module silently failed)
echo "  cmake : $(command -v cmake)  ($(cmake --version 2>/dev/null | head -1))"
echo "  gcc   : $(command -v gcc)    ($(gcc --version   2>/dev/null | head -1))"
echo "  nvcc  : $(command -v nvcc)   ($(nvcc --version  2>/dev/null | tail -1))"

# ─── Banner ──────────────────────────────────────────────────────────────────
echo "==============================================================="
echo "MPM benchmark — job ${SLURM_JOB_ID:-local}  on $(hostname)"
echo "  scene=${SCENE} steps=${STEPS} warmup=${WARMUP} dt=${DT} grid=${GRID}"
echo "  omp sweep: ${OMP_LIST}"
echo "  CSV: ${CSV}"
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 | tr ' ' '_')
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null \
    || echo "  (nvidia-smi not available — GPU run will be skipped if no compiler)"
echo "  GPU tag for CSV: ${GPU_NAME:-none}"
echo "==============================================================="

# ─── 1. Build CUDA flavour (headless) ────────────────────────────────────────
# Fat-binary arch list covers every modern GPU on this cluster.
# CUDA 13 dropped Maxwell/Pascal so we start at 75 (Turing).
#   75 = rtx2080ti, 80 = a100, 86 = rtxa4500, 89 = rtx4000ada, 90 = h100
# Override with CUDA_ARCH=80 sbatch ... for a single-target build.
CUDA_ARCH="${CUDA_ARCH:-75;80;86;89;90}"
build_cuda_dir="${PROJECT_ROOT}/build-bench-cuda"
echo "[build] configuring CUDA build at ${build_cuda_dir}  (arch=${CUDA_ARCH})"
cmake -S "${PROJECT_ROOT}" -B "${build_cuda_dir}" \
      -DCMAKE_BUILD_TYPE=Release \
      -DMPM_ENABLE_CUDA=ON \
      -DMPM_ENABLE_DISPLAY=OFF \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH}"
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
    run_bench "${CUDA_EXE}" "${COMMON_ARGS[@]}" --tag "cuda-${GPU_NAME:-unknown}"
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
