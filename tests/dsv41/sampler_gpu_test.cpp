// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
#include "strata/dsv41/sampler.hpp"
#include "strata/core/readback.hpp"
#include "sycl_test.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <iterator>
#include <limits>
#include <new>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace {
namespace sk = strata::dsv41k;
void check(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

struct Memory {
    sycl::queue& q;
    std::vector<void*> allocated;
    explicit Memory(sycl::queue& queue) : q(queue) {}
    ~Memory() {
        try { q.wait(); for (void* p : allocated) sycl::free(p, q); }
        catch (...) { std::terminate(); }
    }
    Memory(const Memory&) = delete;
    Memory& operator=(const Memory&) = delete;
    template <class T> T* get(size_t size) {
        auto* p = sycl::malloc_device<T>(size, q);
        if (!p) throw std::bad_alloc();
        allocated.push_back(p);
        return p;
    }
};

struct Candidate { int id; float score; };
struct Distribution { std::vector<Candidate> kept; std::vector<double> probabilities; };

// An independent full stable sort is intentionally different from the device's histogram and radix selection.
Distribution reference(const float* logits, int vocab, const std::vector<float>& bias, const int* history,
                       int history_len, const sk::SamplerParams& p) {
    std::vector<Candidate> candidates;
    for (int id = 0; id < vocab; ++id) {
        int count = 0;
        if (history)
            for (int i = std::max(0, history_len - p.penalty_last_n); i < history_len; ++i) count += history[i] == id;
        float score = logits[id] + (bias.empty() ? 0.f : bias[size_t(id)]);
        if (count) {
            score = score <= 0.f ? score * p.penalty_repeat : score / p.penalty_repeat;
            score -= float(count) * p.penalty_freq + p.penalty_present;
        }
        candidates.push_back({id, score});
    }
    std::stable_sort(candidates.begin(), candidates.end(), [](const Candidate& a, const Candidate& b) {
        if (std::isnan(a.score)) return false;
        if (std::isnan(b.score)) return true;
        return a.score > b.score;
    });
    if (p.greedy || p.temperature <= 0.f) return {{candidates[0]}, {1.}};
    const int k = std::min(vocab, p.top_k > 0 && p.top_k < 64 ? p.top_k : 64);
    candidates.resize(size_t(k));
    size_t keep = candidates.size();
    const float maximum = candidates.front().score;
    if (p.top_p < 1.f) {
        double total = 0;
        for (const auto& candidate : candidates) total += std::exp(double(candidate.score) - double(maximum));
        double sum = 0;
        for (size_t i = 0; i < candidates.size(); ++i) {
            sum += std::exp(double(candidates[i].score) - double(maximum)) / total;
            if (sum >= double(p.top_p)) { keep = i + 1; break; }
        }
        keep = std::max(keep, size_t(std::min(k, std::max(0, p.min_keep))));
    }
    if (p.min_p > 0.f) {
        const float threshold = maximum + std::log(p.min_p);
        for (size_t i = 0; i < keep; ++i) if (candidates[i].score < threshold) { keep = i; break; }
    }
    candidates.resize(std::max<size_t>(1, keep));
    const float inverse = 1.f / p.temperature, scaled_maximum = candidates.front().score * inverse;
    double total = 0;
    std::vector<double> probability;
    for (const auto& candidate : candidates) {
        const float scaled = candidate.score * inverse;
        probability.push_back(std::exp(double(scaled) - double(scaled_maximum)));
        total += probability.back();
    }
    for (auto& value : probability) value /= total;
    return {std::move(candidates), std::move(probability)};
}

// Counter-based Philox specified by the sampler API; each row advances the counter exactly once.
float uniform(uint64_t seed, uint64_t counter) {
    uint32_t x[] = {uint32_t(counter), uint32_t(counter >> 32), uint32_t(seed), uint32_t(seed >> 32)};
    for (uint32_t round = 0; round < 10; ++round) {
        const uint64_t a = uint64_t(0x9e3779b9u) * x[0], b = uint64_t(0xbb67ae85u) * x[2];
        const uint32_t next[] = {uint32_t(b >> 32) ^ x[1] ^ round, uint32_t(b), uint32_t(a >> 32) ^ x[3], uint32_t(a)};
        std::copy(std::begin(next), std::end(next), std::begin(x));
    }
    return float(x[0] >> 8) / 16777216.f;
}
int draw(const Distribution& d, float u) {
    double sum = 0;
    for (size_t i = 0; i < d.kept.size(); ++i) {
        sum += d.probabilities[i];
        if (double(u) < sum) return d.kept[i].id;
    }
    return d.kept.back().id;
}

void run_case(const char* name, const std::vector<float>& logits, int vocab, sk::SamplerParams p,
              const std::vector<int>& history = {}, int history_len = 0, const std::vector<float>& bias = {},
              sycl::queue* selected_queue = nullptr) {
    auto& q = selected_queue ? *selected_queue : dsv41_test::queue();
    const size_t tokens = logits.size() / size_t(vocab);
    std::vector<int> output(tokens + 2, -12345), drafts(tokens + 2, -12345);
    std::vector<sk::SpecList> lists(tokens);
    std::vector<Distribution> expected;
    expected.reserve(tokens);
    for (size_t row = 0; row < tokens; ++row)
        expected.push_back(reference(logits.data() + row * size_t(vocab), vocab, bias,
            history.empty() ? nullptr : history.data() + row * size_t(history_len), history_len, p));
    Memory memory(q);  // Complete transfers before any of their host storage leaves scope.
    auto* dl = memory.get<float>(logits.size());
    auto* dout = memory.get<int>(output.size());
    int* dh = nullptr;
    q.memcpy(dl, logits.data(), logits.size() * sizeof(float));
    if (!history.empty()) {
        dh = memory.get<int>(history.size());
        q.memcpy(dh, history.data(), history.size() * sizeof(int));
    }
    if (!bias.empty()) {
        auto* db = memory.get<float>(bias.size());
        q.memcpy(db, bias.data(), bias.size() * sizeof(float));
        p.logit_bias = db;
    }
    q.memcpy(dout, output.data(), output.size() * sizeof(int)).wait_and_throw();
    sk::sample_tokens(dl, int(tokens), vocab, dh, history_len, p, dout + 1, q);
    strata::core::readback(q, output.data(), dout, output.size() * sizeof(int));
    check(output.front() == -12345 && output.back() == -12345, "sampler output guard changed");
    for (size_t row = 0; row < tokens; ++row) {
        const int id = p.greedy || p.temperature <= 0.f ? expected[row].kept.front().id
            : draw(expected[row], uniform(p.seed, p.counter + row));
        check(output[row + 1] == id, "sample_tokens differs from CPU reference");
    }
    if (!p.greedy && p.temperature > 0.f && !p.penalty_last_n) {
        auto* dq = memory.get<sk::SpecList>(tokens);
        auto* ddraft = memory.get<int>(drafts.size());
        q.memcpy(ddraft, drafts.data(), drafts.size() * sizeof(int));
        sk::sample_draft(dl, int(tokens), vocab, p, dq, ddraft + 1, q);
        const strata::core::ReadbackSpan spans[] = {{lists.data(), dq, lists.size() * sizeof(sk::SpecList)},
            {drafts.data(), ddraft, drafts.size() * sizeof(int)}};
        strata::core::readback(q, spans);
        check(drafts.front() == -12345 && drafts.back() == -12345, "draft output guard changed");
        for (size_t row = 0; row < tokens; ++row) {
            const auto& ref = expected[row];
            check(lists[row].n == int(ref.kept.size()), "draft kept count differs");
            for (size_t i = 0; i < ref.kept.size(); ++i) {
                check(lists[row].ids[i] == ref.kept[i].id, "draft candidate ID or tie order differs");
                check(std::isfinite(lists[row].p[i]) && std::abs(lists[row].p[i] - ref.probabilities[i]) <= 1.e-14,
                      "draft probability differs");
            }
            check(drafts[row + 1] == draw(ref, uniform(p.seed ^ 0x5bd1e9955bd1e995ull, p.counter + row)),
                  "draft Philox draw differs");
        }
        // Identical p and q must accept every draft; this also covers per-row SpecList addressing.
        sk::sample_verify(dl, int(tokens), vocab, p, dq, ddraft + 1, dout + 1, q);
        strata::core::readback(q, output.data(), dout, output.size() * sizeof(int));
        check(output == drafts, "verify failed to accept its identical distribution");
    }
    std::printf("sampler: %s OK\n", name);
}
}  // namespace

int main() try {
    if (sycl::device::get_devices(sycl::info::device_type::gpu).empty()) return 77;
    sk::SamplerParams p;
    p.seed = 12345; p.counter = 17; p.top_p = .9f; p.min_p = .01f; p.temperature = .8f;
    constexpr int vocab = 129280;
    std::vector<float> logits(size_t(vocab) * 3);
    for (size_t i = 0; i < logits.size(); ++i) logits[i] = float((i * 1103515245u + 12345u) & 65535u) / 4096.f - 8.f;
    for (size_t row = 0; row < 3; ++row) { logits[row * vocab + 17] = 10.f; logits[row * vocab + 29] = 10.f; }
    run_case("coarse selection / ties", logits, vocab, p);
    p.top_k = 0; p.top_p = 1.f; p.min_p = 0.f; p.temperature = 1.f;
    run_case("flat radix fallback / top_k off", std::vector<float>(vocab, 2.f), vocab, p);
    p.top_k = 1000;
    run_case("top_k above limit", std::vector<float>(73, 2.f), 73, p);
    std::vector<float> small(34);
    std::iota(small.begin(), small.end(), -17.f);
    run_case("vocabulary below 64", small, 17, p);
    p.greedy = true;
    const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
    std::vector<float> edges(51, nan);
    for (size_t i = 18; i < 34; ++i) edges[i] = -inf;
    edges[34 + 2] = -0.f; edges[34 + 3] = +0.f; edges[34 + 4] = -inf;
    run_case("greedy NaN / infinity / signed zero", edges, 17, p);
    p.greedy = false; p.top_k = 20; p.penalty_last_n = 5;
    p.penalty_repeat = 1.25f; p.penalty_freq = .25f; p.penalty_present = .5f;
    std::vector<float> penalised(146), bias(73);
    for (size_t i = 0; i < penalised.size(); ++i) penalised[i] = float(i % 19) * .125f - 1.f;
    bias[17] = .25f; bias[29] = -.25f;
    const std::vector<int> history{0, 0, 0, 0, 17, 17, 29, -1, 1000, 1, 1, 1, 1, 29, 29, 17, -2, 1001};
    run_case("penalty tail / invalid IDs / bias", penalised, 73, p, history, 9, bias);
    p.greedy = true;
    run_case("greedy penalties", penalised, 73, p, history, 9, bias);
    p = sk::SamplerParams{};
    p.top_k = 64; p.top_p = 1.f;
    auto& original = dsv41_test::queue();
    const auto context = original.get_context();
    const auto device = original.get_device();
    {
        sycl::queue second(context, device, sycl::property::queue::in_order{});
        run_case("same context / second queue / larger batch", std::vector<float>(size_t(17) * 7, 2.f), 17, p, {}, 0, {}, &second);
    }
    run_case("after transient queue destruction", small, 17, p);
    {
        const sycl::context other_context(device);
        sycl::queue other(other_context, device, sycl::property::queue::in_order{});
        run_case("different context", small, 17, p, {}, 0, {}, &other);
    }
    run_case("original context again", small, 17, p);
    std::array<std::exception_ptr, 2> errors{};
    {
        std::array<std::jthread, 2> workers;
        for (size_t index = 0; index < workers.size(); ++index)
            workers[index] = std::jthread([&, index] {
                try {
                    sycl::queue independent(context, device, sycl::property::queue::in_order{});
                    auto params = p;
                    params.counter = index;
                    run_case("concurrent host thread", small, 17, params, {}, 0, {}, &independent);
                    params.greedy = true;
                    run_case("concurrent host thread / greedy", small, 17, params, {}, 0, {}, &independent);
                } catch (...) { errors[index] = std::current_exception(); }
            });
    }
    for (const auto& error : errors) if (error) std::rethrow_exception(error);
    std::puts("sampler GPU: all cases passed");
    return 0;
} catch (const std::exception& e) {
    std::fprintf(stderr, "sampler GPU: %s\n", e.what());
    return 1;
}
