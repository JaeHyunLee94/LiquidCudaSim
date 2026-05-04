// ─────────────────────────────────────────────────────────────────────────────
// benchmark.cu — headless MPM benchmark with per-stage timing breakdown
//
// Reports wall-clock time per simulation step plus a breakdown into
//   p2g, updateGrid, g2p,   (and transfer for the CUDA path).
//
// Designed to compare CUDA vs OpenMP performance on HPC nodes.
//
// Usage (one binary handles both paths; backend is chosen at compile time):
//   ./benchmark [--scene cube|bunny] [--steps N] [--warmup W] [--dt DT]
//               [--grid R] [--csv path/to.csv] [--tag LABEL]
//
// Behaviour:
//   • Builds the scene once, runs `warmup` untimed steps, then `steps` timed
//     steps, then prints averages and (optionally) appends a CSV row.
//   • Backend label is "cuda" when MPM_CUDA_AVAILABLE, else "openmp".
//   • OpenMP thread count is whatever OMP_NUM_THREADS the launcher sets.
// ─────────────────────────────────────────────────────────────────────────────

#include <Engine.h>
#include <SimApp.h>
#include <fmt/core.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

struct Args {
  std::string scene  = "cube";
  std::string csv    = "";       // empty → no CSV output
  std::string tag    = "";       // free-form label written into CSV
  unsigned    steps  = 200;
  unsigned    warmup = 20;
  float       dt     = 1e-4f;
  int         grid   = 64;       // grid resolution per axis
};

static void parseArgs(int argc, char **argv, Args &a) {
  for (int i = 1; i < argc; ++i) {
    auto take = [&](const char *flag) -> const char* {
      if (std::strcmp(argv[i], flag) == 0 && i + 1 < argc) return argv[++i];
      return nullptr;
    };
    if      (auto v = take("--scene"))  a.scene  = v;
    else if (auto v = take("--csv"))    a.csv    = v;
    else if (auto v = take("--tag"))    a.tag    = v;
    else if (auto v = take("--steps"))  a.steps  = (unsigned)std::atoi(v);
    else if (auto v = take("--warmup")) a.warmup = (unsigned)std::atoi(v);
    else if (auto v = take("--dt"))     a.dt     = (float)std::atof(v);
    else if (auto v = take("--grid"))   a.grid   = std::atoi(v);
    else fmt::print(stderr, "[bench] ignoring unknown arg: {}\n", argv[i]);
  }
}

static const char *backendName() {
#ifdef MPM_CUDA_AVAILABLE
  return "cuda";
#else
  return "openmp";
#endif
}

static int ompThreads() {
#ifdef _OPENMP
  // omp_get_max_threads() returns what would be used in a parallel region
  return omp_get_max_threads();
#else
  return 1;
#endif
}

static mpm::EngineConfig makeConfig(int gridRes) {
  return mpm::EngineConfig{
      false,
      mpm::MLS,
      mpm::Explicit,
      mpm::Dense,
      mpm::Vec3i(gridRes, gridRes, gridRes),
      1.2f / (float)gridRes,
      0,                       // m_targetFrame: unused (we drive the loop)
      mpm::Device::GPU         // engineStep dispatches by MPM_CUDA_AVAILABLE
  };
}

static void buildScene(mpm::Engine &engine, const std::string &name) {
  const float grid_dx = engine.getEngineConfig().m_gridCellSize;
  mpm::Entity entity;

  if (name == "bunny") {
    entity.loadFromBgeo("../../assets/bunny_1.bgeo");
  } else {
    // Default: dense water cube at the center of the domain.
    const unsigned int res = engine.getEngineConfig().m_gridResolution[0];
    entity.loadCube(mpm::Vec3f(0.5f, 0.5f, 0.5f), 0.6f,
                    (unsigned int)(std::pow(res, 3) / 4));
  }

  engine.setGravity(mpm::Vec3f(0, 0, -9.8f));

  mpm::Particles particles(entity, mpm::MaterialType::WeaklyCompressibleWater,
                           std::pow(grid_dx * 0.5f, 3), 1);
  engine.addParticles(particles);
  engine.makeAosToSOA();
  engine.resume();   // benchmark always advances; no GUI gate here
}

static void runOneStep(mpm::Engine &engine, float dt, mpm::BenchTiming &out) {
#ifdef MPM_CUDA_AVAILABLE
  engine.integrateWithCudaBench(dt, out);
#else
  engine.integrateBench(dt, out);
#endif
}

struct Stats {
  double mean = 0, stdev = 0, p50 = 0, p95 = 0, min_v = 0, max_v = 0;
};

static Stats summarize(std::vector<double> v) {
  Stats s{};
  if (v.empty()) return s;
  std::sort(v.begin(), v.end());
  s.min_v = v.front();
  s.max_v = v.back();
  s.p50   = v[v.size() / 2];
  s.p95   = v[(size_t)(v.size() * 0.95)];
  double sum = 0;
  for (double x : v) sum += x;
  s.mean = sum / v.size();
  double sq = 0;
  for (double x : v) sq += (x - s.mean) * (x - s.mean);
  s.stdev = std::sqrt(sq / v.size());
  return s;
}

int main(int argc, char **argv) {
  Args a;
  parseArgs(argc, argv, a);

  fmt::print("[bench] backend={} scene={} grid={} dt={} warmup={} steps={} omp_threads={}\n",
             backendName(), a.scene, a.grid, a.dt, a.warmup, a.steps, ompThreads());

  // ── Build engine + scene
  mpm::Engine engine(makeConfig(a.grid));
  buildScene(engine, a.scene);
  fmt::print("[bench] particle_count={}\n", engine.getParticleCount());

  // ── Warmup (untimed). Also stabilises caches / JITs CUDA kernels.
  mpm::BenchTiming t;
  for (unsigned i = 0; i < a.warmup; ++i) runOneStep(engine, a.dt, t);

  // ── Timed loop
  std::vector<double> total, p2g, ugrid, g2p, xfer;
  total.reserve(a.steps); p2g.reserve(a.steps);
  ugrid.reserve(a.steps); g2p.reserve(a.steps); xfer.reserve(a.steps);

  auto wall_t0 = std::chrono::steady_clock::now();
  for (unsigned i = 0; i < a.steps; ++i) {
    runOneStep(engine, a.dt, t);
    total.push_back(t.total_ms);
    p2g.push_back(t.p2g_ms);
    ugrid.push_back(t.updateGrid_ms);
    g2p.push_back(t.g2p_ms);
    xfer.push_back(t.transfer_ms);
  }
  auto wall_t1 = std::chrono::steady_clock::now();
  double wall_ms = std::chrono::duration<double, std::milli>(wall_t1 - wall_t0).count();

  Stats sT = summarize(total), sP = summarize(p2g), sU = summarize(ugrid),
        sG = summarize(g2p),   sX = summarize(xfer);

  // ── Pretty print
  auto row = [](const char *name, const Stats &s) {
    fmt::print("  {:<12}  mean={:8.3f}ms  std={:7.3f}  p50={:8.3f}  p95={:8.3f}  min={:8.3f}  max={:8.3f}\n",
               name, s.mean, s.stdev, s.p50, s.p95, s.min_v, s.max_v);
  };
  fmt::print("[bench] per-step timing over {} steps (wall total {:.1f} ms):\n", a.steps, wall_ms);
  row("total",       sT);
  row("p2g",         sP);
  row("updateGrid",  sU);
  row("g2p",         sG);
#ifdef MPM_CUDA_AVAILABLE
  row("transfer",    sX);
#endif

  // Sanity: sum of breakdown means vs total mean
  double sum_means = sP.mean + sU.mean + sG.mean + sX.mean;
  fmt::print("[bench] breakdown_sum/total = {:.3f}  (closer to 1.0 = better attribution)\n",
             sT.mean > 0 ? sum_means / sT.mean : 0.0);

  // ── CSV append (one row per run; header written when file is new)
  if (!a.csv.empty()) {
    bool needs_header = false;
    {
      std::ifstream probe(a.csv);
      needs_header = !probe.good();
    }
    std::ofstream f(a.csv, std::ios::app);
    if (needs_header) {
      f << "tag,backend,scene,grid,particles,steps,warmup,dt,omp_threads,"
           "total_mean_ms,total_stdev_ms,total_p50_ms,total_p95_ms,"
           "p2g_mean_ms,updateGrid_mean_ms,g2p_mean_ms,transfer_mean_ms,"
           "wall_total_ms\n";
    }
    f << a.tag << ',' << backendName() << ',' << a.scene << ',' << a.grid
      << ',' << engine.getParticleCount() << ',' << a.steps << ',' << a.warmup
      << ',' << a.dt << ',' << ompThreads() << ','
      << sT.mean << ',' << sT.stdev << ',' << sT.p50 << ',' << sT.p95 << ','
      << sP.mean << ',' << sU.mean << ',' << sG.mean << ',' << sX.mean << ','
      << wall_ms << '\n';
    fmt::print("[bench] appended row to {}\n", a.csv);
  }

  return EXIT_SUCCESS;
}
