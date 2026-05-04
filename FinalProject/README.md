# Final Course Project for CS759

CUDA + OpenMP based liquid simulation engine using the Material Point Method (MLS-MPM).

## Build

Vanilla build (auto-detects CUDA, OpenMP, OpenGL):
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Headless build (no OpenGL — required on HPC nodes):
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DMPM_ENABLE_CUDA=ON -DMPM_ENABLE_DISPLAY=OFF
cmake --build build -j
```

OpenMP-only build (no CUDA toolkit needed; useful on a laptop):
```
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DMPM_ENABLE_CUDA=OFF -DMPM_ENABLE_DISPLAY=OFF
cmake --build build -j
```

The same source tree composes into all three modes via `MPM_ENABLE_CUDA` and `MPM_ENABLE_DISPLAY`. When CUDA is missing it silently falls back to the OpenMP path.

## Running scenes

Scenes load assets via relative paths (`../../assets/*.bgeo`, `../../src/render/shader/*.glsl`), so run them **from the `build/scene/` directory**:

```
cd build/scene
./water_fall_bunny     # interactive
./benchmark            # headless benchmark, see flags below
```

## Benchmark

`scene/benchmark.cu` runs a headless MPM step loop and reports per-stage timings (P2G / updateGrid / G2P / H2D+D2H) using `cudaEvent` (GPU) or `std::chrono` (CPU).

```
./benchmark --scene cube --steps 200 --warmup 20 --dt 1e-4 \
            --grid 64 --csv ../../results/local.csv --tag mytag
```

All flags are optional; defaults reproduce the configuration used in the report.

## HPC (UW–Madison Euler)

`scripts/bench.slurm` builds both the CUDA and OpenMP flavours, runs the CUDA benchmark once, then sweeps OpenMP across 1, 2, 4, 8, 16 threads, appending one CSV row per configuration to `results/bench-<jobid>.csv`.

```
sbatch scripts/bench.slurm
# override knobs at submit time:
SCENE=bunny STEPS=500 OMP_LIST="1 2 4 8 16" sbatch scripts/bench.slurm
# pin a GPU type:
sbatch --gres=gpu:a100:1 scripts/bench.slurm
```

## Features
- [x] [MLS-MPM](https://yzhu.io/publication/mpmmls2018siggraph/paper.pdf)
- [x] CUDA backend
- [x] OpenMP backend (auto-fallback when CUDA is unavailable)
- [x] OpenGL renderer + ImGui GUI (interactive mode)
- [x] Headless / HPC mode with per-stage timing breakdown
- [x] SLURM job for reproducible benchmarks
- [x] Mesh importer (.bgeo / .obj)
- [ ] Mesh exporter
- [ ] Mesh obstacle
- [ ] 2D cloth, hair

## Layout

```
FinalProject/
├── src/engine/        solver core (CPU + CUDA)
├── src/render/        OpenGL + ImGui front-end
├── scene/             one .cu per scene + benchmark.cu
├── assets/            .bgeo / .obj inputs
├── external/          vendored deps (Eigen, fmt, partio, GLFW, GLEW, ImGui, …)
├── test/              third-party smoke tests
└── scripts/
    ├── bench.slurm    HPC benchmark job
    └── fill_report.py report generator
```

Run results land in `FinalProject/results/`; SLURM logs land at `FinalProject/bench-<jobid>.{out,err}`.

## Sample Scenes

<img src="https://user-images.githubusercontent.com/46246202/193277677-44a30116-0ef4-4859-960b-5a1984309a23.gif" width="35%" height="35%"/>
<img src="https://user-images.githubusercontent.com/46246202/193274175-186af6bd-3afb-42ae-a073-0b05b8167c7d.gif" width="35%" height="35%"/>
<img src="https://user-images.githubusercontent.com/46246202/193274094-fcbeb376-9767-4972-a11a-f7e301218623.gif" width="35%" height="35%"/>

## Dependencies

All third-party libraries are vendored under `external/` (Eigen, fmt, partio, GLFW, GLEW, ImGui, CompactNSearch, SDFGen, tinyobjloader). No external installs required beyond a CUDA toolkit (optional) and a C++17 compiler.
