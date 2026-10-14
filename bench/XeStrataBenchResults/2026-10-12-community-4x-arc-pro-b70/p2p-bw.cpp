// Measure device-to-device (P2P) copy bandwidth between the Arc Pro B70s, and compare it with the
// host-staged path XeStrata uses today for its layer hand-offs (device -> host -> device).
// Read-only w.r.t. the engine: it allocates its own buffers and frees them.
#include <sycl/sycl.hpp>
#include <chrono>
#include <cstdio>
#include <vector>

using namespace sycl;

static double secs(std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
  return std::chrono::duration<double>(b - a).count();
}

int main(int argc, char **argv) {
  const size_t N = (argc > 1 ? (size_t) atoll(argv[1]) : 256ull) << 20;   // MiB
  std::vector<device> devs;
  for (auto &p : platform::get_platforms()) {
    if (p.get_backend() != backend::ext_oneapi_level_zero) continue;
    for (auto &d : p.get_devices(info::device_type::gpu)) devs.push_back(d);
  }
  printf("level-zero GPUs: %zu   buffer: %zu MiB\n", devs.size(), N >> 20);
  if (devs.size() < 2) { printf("need two devices\n"); return 1; }

  printf("\n  %-12s %-14s %-14s %-12s\n", "pair", "P2P GB/s", "staged GB/s", "speedup");
  for (size_t i = 0; i < devs.size(); ++i) {
    for (size_t j = 0; j < devs.size(); ++j) {
      if (i == j) continue;
      bool can = false;
      try { can = devs[j].ext_oneapi_can_access_peer(devs[i]); } catch (const std::exception &e) {
        printf("  dev%zu->dev%zu: can_access_peer threw: %s\n", i, j, e.what()); continue;
      }
      if (!can) { printf("  dev%zu->dev%zu: no peer access\n", i, j); continue; }
      try {
        devs[j].ext_oneapi_enable_peer_access(devs[i]);
        queue qi{devs[i]}, qj{devs[j]};
        auto *a = malloc_device<char>(N, qi);        // lives on dev i
        auto *b = malloc_device<char>(N, qj);        // lives on dev j
        auto *host = malloc_host<char>(N, qj);       // host staging
        qi.memset(a, 1, N).wait();
        qj.memset(b, 0, N).wait();

        // one warm-up each, then the timed run
        qj.memcpy(b, a, N).wait();
        auto t0 = std::chrono::steady_clock::now();
        qj.memcpy(b, a, N).wait();
        auto t1 = std::chrono::steady_clock::now();
        double p2p = N / secs(t0, t1) / 1e9;

        qj.memcpy(host, a, N).wait();
        qj.memcpy(b, host, N).wait();
        auto t2 = std::chrono::steady_clock::now();
        qj.memcpy(host, a, N).wait();
        qj.memcpy(b, host, N).wait();
        auto t3 = std::chrono::steady_clock::now();
        double staged = N / secs(t2, t3) / 1e9;

        printf("  dev%zu->dev%zu   %8.2f       %8.2f       %6.2fx\n", i, j, p2p, staged, p2p / staged);
        free(a, qi); free(b, qj); free(host, qj);
      } catch (const std::exception &e) {
        printf("  dev%zu->dev%zu: %s\n", i, j, e.what());
      }
    }
  }
  return 0;
}
