#include "strata/core/uma_memory.hpp"
#include <cassert>
#include <limits>
using namespace strata::core;
int main() {
    constexpr uint64_t G = uint64_t(1) << 30;
    assert(parse_uma_gib_env("0", 6) == 0 && parse_uma_gib_env("1024", 6) == 1024);
    for (auto s : {"-1", "1025", "3x", "", " 2"}) assert(parse_uma_gib_env(s, 6) == 6);
    assert(uma_commit_margin_bytes(64*G) == 8*G);
    assert(uma_commit_margin_bytes(100*G) == 10*G);
    assert(uma_commit_margin_bytes(100*G, "2") == 2*G);
    UmaMemoryInput i; i.runtime_total=100; i.runtime_free_known=true; i.runtime_free=90; i.os_available=60; i.headroom=10;
    auto r=compute_uma_free(i); assert(r.effective_free==50 && r.os_capped && !r.commit_capped);
    i.runtime_free_known=false; i.gpu_reclaimable=100; assert(compute_uma_free(i).effective_free==100);
    i.os_available=20; i.gpu_reclaimable=0; assert(compute_uma_free(i).effective_free==10); // cgroup room
    i.os_available=100; i.has_commit_limit=true; i.commit_available=30; i.commit_margin=10;
    r=compute_uma_free(i); assert(r.effective_free==20 && r.commit_capped);
    i.os_available=std::numeric_limits<uint64_t>::max(); i.gpu_reclaimable=1; i.has_commit_limit=false;
    assert(compute_uma_free(i).effective_free==100);
}
