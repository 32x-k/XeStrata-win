// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// src/image/qi_sdpa_cudnn.cpp - see qi_sdpa.hpp: cuDNN's SDPA through cudnn-frontend on NVIDIA GPUs (contrib-llvm),
// cuDNN opened at the first attention (the user's installation, not free software: open_cudnn), on the queue's CUDA
// stream, a graph built once a shape and kept.

#include <cudnn_frontend.h>   // with its dynamic loading: windows.h on Windows
#include <sycl/backend/cuda.hpp>
#if defined(_WIN32)
#include <filesystem>
namespace cudnn_frontend {
HMODULE cudnn_dlhandle = nullptr;   // cudnn_frontend_shim.h's: cuDNN opened at the first attention
}
#else
#include <dlfcn.h>
namespace cudnn_frontend {
void* cudnn_dlhandle = nullptr;
}
#endif

#include "qi_sdpa.hpp"

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

namespace strata::image::sdpa {

namespace {

std::mutex g_mu;

namespace fe = cudnn_frontend;

struct Cudnn {
    bool failed = false;
    cudnnHandle_t handle = nullptr;
    std::map<std::tuple<int64_t, int64_t, int64_t, int64_t>, std::shared_ptr<fe::graph::Graph>> graphs;
    void* ws = nullptr;
    int64_t ws_bytes = 0;
};
std::unordered_map<sycl::device, Cudnn> g_cudnn;

constexpr int64_t kQ = 1, kK = 2, kV = 3, kO = 4;   // the tensors' ids in the variant pack

std::shared_ptr<fe::graph::Graph> cudnn_graph(cudnnHandle_t handle, int64_t rows, int64_t T, int64_t heads,
                                              int64_t hd) {
    auto g = std::make_shared<fe::graph::Graph>();
    g->set_io_data_type(fe::DataType_t::HALF)
        .set_intermediate_data_type(fe::DataType_t::FLOAT)
        .set_compute_data_type(fe::DataType_t::FLOAT);
    const int64_t row = heads * hd;
    auto Q = g->tensor(fe::graph::Tensor_attributes().set_name("Q").set_uid(kQ).set_dim({1, heads, rows, hd}).set_stride(
        {rows * row, hd, row, 1}));
    auto K = g->tensor(fe::graph::Tensor_attributes().set_name("K").set_uid(kK).set_dim({1, heads, T, hd}).set_stride(
        {T * row, hd, row, 1}));
    auto V = g->tensor(fe::graph::Tensor_attributes().set_name("V").set_uid(kV).set_dim({1, heads, T, hd}).set_stride(
        {T * row, hd, row, 1}));
    auto opts = fe::graph::SDPA_attributes().set_name("sdpa").set_generate_stats(false).set_attn_scale(
        1.0f / std::sqrt((float) hd));
    auto [O, stats] = g->sdpa(Q, K, V, opts);
    (void) stats;
    O->set_output(true).set_dim({1, heads, rows, hd}).set_stride({rows * row, hd, row, 1}).set_uid(kO);
    if (!g->build(handle, {fe::HeurMode_t::A}).is_good()) throw std::runtime_error("cuDNN builds no SDPA plan");
    return g;
}

// cuDNN's library opened and a handle made: STRATA_CUDNN_LIB, else the system's (libcudnn.so.9; on Windows
// cudnn64_9.dll on the PATH, then NVIDIA's installer's folders, newest first, then the one pip put in the Python the 7z
// carries, nvidia-cudnn-cu12), the first that makes a handle
void open_cudnn(cudnnHandle_t& handle) {
    if (fe::cudnn_dlhandle != nullptr) {
        if (fe::detail::create_handle(&handle) != CUDNN_STATUS_SUCCESS) throw std::runtime_error("cudnnCreate failed");
        return;
    }
    std::vector<std::string> libs;
    if (const char* lib = std::getenv("STRATA_CUDNN_LIB")) libs.emplace_back(lib);
#if defined(_WIN32)
    namespace fs = std::filesystem;
    libs.emplace_back("cudnn64_9.dll");
    std::error_code ec;
    std::vector<fs::path> found;
    const char* pf = std::getenv("ProgramFiles");
    const fs::path root = fs::path(pf != nullptr ? pf : "C:\\Program Files") / "NVIDIA" / "CUDNN";
    for (const auto& v : fs::directory_iterator(root, ec))
        for (const auto& b : fs::recursive_directory_iterator(v.path() / "bin", ec))
            if (b.path().filename() == "cudnn64_9.dll") found.push_back(b.path());
    std::sort(found.begin(), found.end(), std::greater<>());
    for (const auto& f : found) libs.push_back(f.string());
    char exe[MAX_PATH] = {};
    fs::path nvidia;   // <7z>\python\Lib\site-packages\nvidia, beside <7z>\app\engine\strata-image.exe
    if (GetModuleFileNameA(nullptr, exe, MAX_PATH) > 0) {
        nvidia = fs::path(exe).parent_path().parent_path().parent_path() / "python" / "Lib" / "site-packages" /
                 "nvidia";
        const fs::path pip = nvidia / "cudnn" / "bin" / "cudnn64_9.dll";
        if (fs::exists(pip, ec)) libs.push_back(pip.string());
    }
    // cudnn-frontend opens the CUDA runtime by its Linux names: on Windows cudart64_12.dll, as the user installed it
    // (the PATH, the CUDA toolkit's, the 7z's Python's nvidia-cuda-runtime-cu12), unless CUDNN_FRONTEND_CUDART_LIB_NAME
    // names one
    if (std::getenv("CUDNN_FRONTEND_CUDART_LIB_NAME") == nullptr) {
        std::vector<fs::path> dirs;
        if (const char* path = std::getenv("PATH")) {
            for (std::string rest = path; !rest.empty();) {
                const size_t k = rest.find(';');
                if (k != 0) dirs.emplace_back(rest.substr(0, k));
                rest = k == std::string::npos ? "" : rest.substr(k + 1);
            }
        }
        if (const char* c = std::getenv("CUDA_PATH")) dirs.push_back(fs::path(c) / "bin");
        if (!nvidia.empty()) dirs.push_back(nvidia / "cuda_runtime" / "bin");
        for (const fs::path& d : dirs)
            if (fs::exists(d / "cudart64_12.dll", ec)) {
                _putenv_s("CUDNN_FRONTEND_CUDART_LIB_NAME", (d / "cudart64_12.dll").string().c_str());
                break;
            }
    }
#else
    libs.emplace_back("libcudnn.so.9");
    // cudnn-frontend opens the first of libcudart.so.12 and .13 it finds: the CUDA version the engine was built with
    // (cuDNN's for it on the library path, setup.py's cudnn_lib_dirs), not another a PC also has
    setenv("CUDNN_FRONTEND_CUDART_LIB_NAME", ("libcudart.so." + std::to_string(CUDART_VERSION / 1000)).c_str(), 0);
#endif
    std::string why = "cuDNN is not installed";
    for (const std::string& lib : libs) {
#if defined(_WIN32)
        // its other DLLs beside it: those it links, and those cudnn64_9.dll loads itself later (cudnn_graph64_9.dll and
        // the engines), which Windows looks for on the PATH
        if (lib.find('\\') != std::string::npos) {
            const char* old = std::getenv("PATH");
            const std::string path = fs::path(lib).parent_path().string() + ";" + (old != nullptr ? old : "");
            _putenv_s("PATH", path.c_str());
        }
        HMODULE h = LoadLibraryExA(lib.c_str(), nullptr, lib.find('\\') != std::string::npos ?
                                                             LOAD_WITH_ALTERED_SEARCH_PATH : 0);
#else
        void* h = dlopen(lib.c_str(), RTLD_NOW | RTLD_GLOBAL);
#endif
        if (h == nullptr) continue;
        fe::cudnn_dlhandle = h;
        if (fe::detail::create_handle(&handle) == CUDNN_STATUS_SUCCESS) return;
        why = lib + ": cudnnCreate failed";
#if defined(_WIN32)
        FreeLibrary(h);
#else
        dlclose(h);
#endif
        fe::cudnn_dlhandle = nullptr;
    }
    throw std::runtime_error(why);
}

}  // namespace

bool cudnn(sycl::queue& q, const uint16_t* q16, const uint16_t* k16, const uint16_t* v16, int64_t q0, int64_t q1,
                int64_t T, int64_t heads, int64_t hd, uint16_t* out16) {
    std::lock_guard<std::mutex> lock(g_mu);
    Cudnn& d = g_cudnn[q.get_device()];
    if (d.failed) return false;
    try {
        if (d.handle == nullptr) open_cudnn(d.handle);
        const CUstream cs = sycl::get_native<sycl::backend::ext_oneapi_cuda>(q);
        if (fe::detail::set_stream(d.handle, cs) != CUDNN_STATUS_SUCCESS) throw std::runtime_error("cudnnSetStream");
        const auto key = std::make_tuple(q1 - q0, T, heads, hd);
        auto it = d.graphs.find(key);
        if (it == d.graphs.end()) it = d.graphs.emplace(key, cudnn_graph(d.handle, q1 - q0, T, heads, hd)).first;
        int64_t ws = 0;
        if (!it->second->get_workspace_size(ws).is_good()) throw std::runtime_error("cuDNN's workspace");
        if (ws > d.ws_bytes) {
            q.wait();
            if (d.ws != nullptr) sycl::free(d.ws, q);
            d.ws = sycl::malloc_device(ws, q);
            d.ws_bytes = d.ws != nullptr ? ws : 0;
        }
        const int64_t row = heads * hd;
        std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> pack = {
            {kQ, const_cast<uint16_t*>(q16 + q0 * row)}, {kK, const_cast<uint16_t*>(k16)},
            {kV, const_cast<uint16_t*>(v16)}, {kO, out16}};
        if (!it->second->execute(d.handle, pack, d.ws).is_good()) throw std::runtime_error("cuDNN's SDPA failed");
        return true;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-image: cuDNN's attention is not taken on this GPU (%s): the engine's own kernel\n",
                     e.what());
        d.failed = true;
        return false;
    }
}

}  // namespace strata::image::sdpa
