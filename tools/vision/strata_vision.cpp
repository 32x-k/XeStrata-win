// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
// tools/vision/strata_vision.cpp - the image half of Strata's multimodal path.
//
// Turns an image into the embeddings the text model reads in place of its <|image_pad|> tokens, with llama.cpp's
// mtmd library and the model's mmproj file (vision encoder + projector).  The engine does the rest: it places the
// rows at the image's pad tokens and gives them their 2-D M-RoPE positions (see --serve GENI in generate.cpp).
//
//   strata-vision --mmproj <mmproj.gguf> --model <text model .gguf, first split> [--gpu [--gpu-pci ADDR]]
//                 [--threads N] [--max-tokens N] [--min-tokens N] [--flash-attn on|off|auto]
//
// --gpu-pci takes the GPU whose PCI address (domain:bus:device.function) is ADDR; without it, mtmd takes the backend's
// first GPU, which with Vulkan can be the CPU's integrated graphics or another vendor's card.
// Flash attention defaults to auto, and to off on the CPU when ggml is built with AVX-512 (see main).
//
// Resident: prints "READY <n_embd>", then per stdin line
//   ENC <image path> <output path>   ->  "OK <n_tokens> <nx> <ny> <ms>"  or  "ERR <message>"
//   QUIT
// The output file is  int32 {0x31455653 'SVE1', n_tokens, nx, ny, n_embd}  then float32 [n_tokens][n_embd],
// row i at grid position (x = i % nx, y = i / nx).  The text model is opened vocab-only (no weights).
#include "ggml-cpu.h"
#include "gguf.h"
#include "gguf_meta.h"
#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <algorithm>
#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>
#if defined(_WIN32) && defined(STRATA_VISION_VULKAN)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <winternl.h>
#include <d3dkmthk.h>
#include <vulkan/vulkan.h>
#endif

namespace {

#if defined(_WIN32) && defined(STRATA_VISION_VULKAN)
// Windows: a driver without VK_EXT_pci_bus_info (AMD's) gives ggml no PCI address, so --gpu-pci finds no device.  The
// adapter at that PCI address is found by its LUID instead (D3DKMT's adapter address), then the Vulkan device with the
// same deviceLUID; its index in the Vulkan instance's device list is what GGML_VK_VISIBLE_DEVICES takes.  -1: none.
int vk_index_by_pci(const std::string& pci) {
    unsigned dom = 0, bus = 0, dev = 0, fn = 0;
    if (std::sscanf(pci.c_str(), "%x:%x:%x.%x", &dom, &bus, &dev, &fn) != 4) return -1;
    D3DKMT_ENUMADAPTERS2 en{};
    if (D3DKMTEnumAdapters2(&en) != 0 || en.NumAdapters == 0) return -1;
    std::vector<D3DKMT_ADAPTERINFO> ad(en.NumAdapters);
    en.pAdapters = ad.data();
    if (D3DKMTEnumAdapters2(&en) != 0) return -1;
    LUID luid{};
    bool found = false;
    for (ULONG i = 0; i < en.NumAdapters; ++i) {
        D3DKMT_ADAPTERADDRESS addr{};
        D3DKMT_QUERYADAPTERINFO q{};
        q.hAdapter = ad[i].hAdapter;
        q.Type = KMTQAITYPE_ADAPTERADDRESS;
        q.pPrivateDriverData = &addr;
        q.PrivateDriverDataSize = sizeof addr;
        if (!found && D3DKMTQueryAdapterInfo(&q) == 0 && addr.BusNumber == bus && addr.DeviceNumber == dev &&
            addr.FunctionNumber == fn) {
            luid = ad[i].AdapterLuid;
            found = true;
        }
        D3DKMT_CLOSEADAPTER c{ad[i].hAdapter};
        D3DKMTCloseAdapter(&c);
    }
    if (!found) return -1;
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.apiVersion = VK_API_VERSION_1_1;
    VkInstanceCreateInfo ci{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ci.pApplicationInfo = &app;
    VkInstance inst = VK_NULL_HANDLE;
    if (vkCreateInstance(&ci, nullptr, &inst) != VK_SUCCESS) return -1;
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(inst, &n, nullptr);
    std::vector<VkPhysicalDevice> pd(n);
    vkEnumeratePhysicalDevices(inst, &n, pd.data());
    int index = -1;
    for (uint32_t i = 0; i < n && index < 0; ++i) {
        VkPhysicalDeviceIDProperties id{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ID_PROPERTIES};
        VkPhysicalDeviceProperties2 p{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
        p.pNext = &id;
        vkGetPhysicalDeviceProperties2(pd[i], &p);
        if (id.deviceLUIDValid && std::memcmp(id.deviceLUID, &luid, sizeof luid) == 0) index = (int) i;
    }
    vkDestroyInstance(inst, nullptr);
    return index;
}
#endif

void quiet_log(ggml_log_level level, const char* text, void*) {
    if (level >= GGML_LOG_LEVEL_WARN) std::fputs(text, stderr);
}

// "ENC <image> <out>": paths may contain spaces, so the image path ends at the last " " before the output path;
// the server always writes paths without spaces, but a user calling the tool by hand might not
bool parse_enc(const std::string& line, std::string& img, std::string& out) {
    if (line.rfind("ENC ", 0) != 0) return false;
    const std::string rest = line.substr(4);
    const size_t sp = rest.rfind(' ');
    if (sp == std::string::npos || sp == 0) return false;
    img = rest.substr(0, sp);
    out = rest.substr(sp + 1);
    return !img.empty() && !out.empty();
}

}  // namespace

int main(int argc, char** argv) {
    std::string mmproj, model;
    bool gpu = false;
    std::string gpu_pci;
    int threads = 0, max_tokens = 0, min_tokens = 0;
    llama_flash_attn_type fa = LLAMA_FLASH_ATTN_TYPE_AUTO;
    bool fa_given = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { std::fprintf(stderr, "%s needs a value\n", a.c_str()); std::exit(2); }
            return argv[++i];
        };
        if (a == "--mmproj") mmproj = next();
        else if (a == "--model") model = next();
        else if (a == "--gpu") gpu = true;
        else if (a == "--gpu-pci") gpu_pci = next();
        else if (a == "--threads") threads = std::atoi(next().c_str());
        else if (a == "--max-tokens") max_tokens = std::atoi(next().c_str());
        else if (a == "--min-tokens") min_tokens = (int) std::strtol(next().c_str(), nullptr, 10);   // mtmd image_min_tokens (upstream #767)
        else if (a == "--flash-attn") {   // FA keeps K and V in FP16; off = the attention in FP32
            const std::string v = next();
            fa = v == "on" ? LLAMA_FLASH_ATTN_TYPE_ENABLED : v == "off" ? LLAMA_FLASH_ATTN_TYPE_DISABLED
                                                                        : LLAMA_FLASH_ATTN_TYPE_AUTO;
            fa_given = true;
        }
        else { std::fprintf(stderr, "unknown argument %s\n", a.c_str()); return 2; }
    }
    if (mmproj.empty() || model.empty()) {
        std::fprintf(stderr, "usage: strata-vision --mmproj <mmproj.gguf> --model <model.gguf> [--gpu [--gpu-pci ADDR]] "
                             "[--threads N] [--max-tokens N] [--min-tokens N] [--flash-attn on|off|auto]\n");
        return 2;
    }
    llama_log_set(quiet_log, nullptr);
    mtmd_helper_log_set(quiet_log, nullptr);
#if defined(_WIN32) && defined(STRATA_VISION_VULKAN)
    // before the backends start: show ggml only the card at --gpu-pci (see vk_index_by_pci), unless the person chose
    // the devices with GGML_VK_VISIBLE_DEVICES; mtmd then takes that one card as its first GPU
    bool pci_by_luid = false;
    if (gpu && !gpu_pci.empty() && std::getenv("GGML_VK_VISIBLE_DEVICES") == nullptr) {
        const int vk = vk_index_by_pci(gpu_pci);
        if (vk >= 0 && _putenv_s("GGML_VK_VISIBLE_DEVICES", std::to_string(vk).c_str()) == 0) pci_by_luid = true;
        if (pci_by_luid)
            std::fprintf(stderr, "strata-vision: the GPU at %s by its adapter LUID (Vulkan device %d)\n",
                         gpu_pci.c_str(), vk);
    }
#endif
    llama_backend_init();

    const auto load_t0 = std::chrono::steady_clock::now();
    // ask the OS for the projector file up front (128 KiB steps) so the loader's sequential reads find it cached;
    // STRATA_READ_AHEAD=0 turns it off, as in the engine.  Windows has no posix_fadvise: its cache manager reads
    // ahead of the loader's sequential reads by itself
#if !defined(_WIN32)
    if (const char* ra = std::getenv("STRATA_READ_AHEAD"); ra == nullptr || std::strtol(ra, nullptr, 10) != 0) {
        const int fd = open(mmproj.c_str(), O_RDONLY);
        if (fd >= 0) {
            const off_t n = lseek(fd, 0, SEEK_END);
            for (off_t at = 0; at < n; at += (off_t) (128 << 10))
                (void) posix_fadvise(fd, at, std::min<off_t>((off_t) (128 << 10), n - at), POSIX_FADV_WILLNEED);
            close(fd);
        }
    }
#endif
    // the vocabulary and header only: the weights (and their tensor types, which can be Strata-private ones such as
    // Gyro's TQK6 / TQK7) are never touched - see gguf_meta.h
    std::string how;
    llama_model* text = strata_open_text_model(model, &how);
    if (!text) { std::printf("ERR cannot open the text model %s (%s)\n", model.c_str(), how.c_str()); std::fflush(stdout); return 1; }

    mtmd_context_params cp = mtmd_context_params_default();
    cp.use_gpu = gpu;
    if (gpu && !gpu_pci.empty()) {
        for (size_t i = 0; i < ggml_backend_dev_count() && cp.device == nullptr; ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            ggml_backend_dev_props props;
            ggml_backend_dev_get_props(d, &props);
            if (props.device_id != nullptr && gpu_pci == props.device_id) cp.device = d;
        }
#if defined(_WIN32) && defined(STRATA_VISION_VULKAN)
        // no PCI address from the driver: the one card left visible is the one at --gpu-pci
        for (size_t i = 0; pci_by_luid && i < ggml_backend_dev_count() && cp.device == nullptr; ++i) {
            ggml_backend_dev_t d = ggml_backend_dev_get(i);
            const enum ggml_backend_dev_type t = ggml_backend_dev_type(d);
            if (t == GGML_BACKEND_DEVICE_TYPE_GPU || t == GGML_BACKEND_DEVICE_TYPE_IGPU) cp.device = d;
        }
#endif
        if (cp.device == nullptr) {
            std::printf("ERR no GPU at PCI address %s\n", gpu_pci.c_str());
            std::fflush(stdout);
            return 1;
        }
    }
    cp.print_timings = false;
    cp.warmup = false;
    // On the CPU "auto" turns flash attention on.  ggml's fast (tiled) CPU kernel for it needs the head size, 72 in this
    // encoder, to be a multiple of the vector width: 8 floats with AVX2, 16 with AVX-512, where ggml falls back to a
    // kernel that is several times slower and accumulates in FP16 (upstream e7ec6b1a, a Ryzen 7 7700X: AVX2 8-10 s with
    // it, 14-15 s without; AVX-512 44 s with it, 13 s without, for a 1024x1024 picture)
    if (!gpu && !fa_given && ggml_cpu_has_avx512()) fa = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.flash_attn_type = fa;
    // on the CPU without --threads: one per core (mtmd's own default is 4 threads)
    if (threads <= 0 && !gpu) threads = (int) std::max(1u, std::thread::hardware_concurrency() / 2);
    if (threads > 0) cp.n_threads = threads;
    if (max_tokens > 0) cp.image_max_tokens = max_tokens;
    if (min_tokens > 0) cp.image_min_tokens = min_tokens;
    mtmd_context* ctx = mtmd_init_from_file(mmproj.c_str(), text, cp);
    if (!ctx || !mtmd_support_vision(ctx)) {
        std::printf("ERR cannot load the vision encoder %s\n", mmproj.c_str());
        std::fflush(stdout);
        return 1;
    }
    std::fprintf(stderr, "strata-vision: model files loaded in %.1f s\n",
                 std::chrono::duration<double>(std::chrono::steady_clock::now() - load_t0).count());
    // a vocab-only model reports no hparams, so the width comes from the projector: its output is the model's input
    int n_embd = 0;
    {
        gguf_init_params gp{true, nullptr};
        gguf_context* gg = gguf_init_from_file(mmproj.c_str(), gp);
        const int64_t k = gg ? gguf_find_key(gg, "clip.vision.projection_dim") : -1;
        if (k >= 0) n_embd = (int) gguf_get_val_u32(gg, k);
        if (gg) gguf_free(gg);
    }
    if (n_embd <= 0) { std::printf("ERR the vision encoder has no projection_dim\n"); std::fflush(stdout); return 1; }
    // Warm up at the LARGEST picture before READY: the encoder's GPU work buffers are allocated now, not at the first
    // real picture.  The server starts this process before the engine, so the engine sizes its expert cache from
    // what is really left. (A square image well above any cap; mtmd scales it to the token limit.)
    // On the CPU there is no VRAM to reserve, and the warm-up would only delay the engine's start by one encode
    // (~6 s at 1,024 tokens).
    if (!gpu) std::fprintf(stderr, "strata-vision: on the CPU, %d threads, no warm-up\n", threads);
    else {
        const auto warm_t0 = std::chrono::steady_clock::now();
        const uint32_t side = 2048;
        std::vector<unsigned char> rgb((size_t) side * side * 3, 128);
        mtmd_bitmap* bm = mtmd_bitmap_init(side, side, rgb.data());
        mtmd_input_chunks* chunks = mtmd_input_chunks_init();
        const std::string marker = mtmd_default_marker();
        mtmd_input_text txt{marker.c_str(), marker.size(), false, true};
        const mtmd_bitmap* bms[1] = {bm};
        int warm_tokens = 0;
        if (bm && mtmd_tokenize(ctx, chunks, &txt, bms, 1) == 0) {
            for (size_t c = 0; c < mtmd_input_chunks_size(chunks); ++c) {
                const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks, c);
                if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE && mtmd_encode_chunk(ctx, ch) == 0)
                    warm_tokens = (int) mtmd_input_chunk_get_n_tokens(ch);
            }
        }
        std::fprintf(stderr, "strata-vision: warmed up at %d image tokens in %.1f s\n", warm_tokens,
                     std::chrono::duration<double>(std::chrono::steady_clock::now() - warm_t0).count());
        mtmd_input_chunks_free(chunks);
        if (bm) mtmd_bitmap_free(bm);
        // READY tells the server the encoder's buffers are allocated, and it starts the engine on what is left: an
        // encoder whose warm-up failed has not allocated them and would fail its first picture anyway
        if (warm_tokens <= 0) {
            std::printf("ERR the warm-up encode failed\n");
            std::fflush(stdout);
            return 1;
        }
    }
    std::printf("READY %d\n", n_embd);
    std::fflush(stdout);

    std::string line;
    while (std::getline(std::cin, line)) {
        while (!line.empty() && (line.back() == '\r' || line.back() == '\n')) line.pop_back();
        if (line == "QUIT") break;
        std::string img, out;
        if (!parse_enc(line, img, out)) { std::printf("ERR expected: ENC <image> <output>\n"); std::fflush(stdout); continue; }
        const auto t0 = std::chrono::steady_clock::now();
        mtmd_helper_bitmap_wrapper bw = mtmd_helper_bitmap_init_from_file(ctx, img.c_str(), false,
                                                                            mtmd_helper_init_opt_default());
        if (!bw.bitmap) { std::printf("ERR cannot read the image %s\n", img.c_str()); std::fflush(stdout); continue; }
        mtmd_input_chunks* chunks = mtmd_input_chunks_init();
        const std::string prompt = mtmd_default_marker();
        mtmd_input_text txt{prompt.c_str(), prompt.size(), false, true};
        const mtmd_bitmap* bm[1] = {bw.bitmap};
        std::string err;
        const mtmd_input_chunk* ichunk = nullptr;
        if (mtmd_tokenize(ctx, chunks, &txt, bm, 1) != 0) err = "the image could not be preprocessed";
        for (size_t c = 0; err.empty() && c < mtmd_input_chunks_size(chunks); ++c) {
            const mtmd_input_chunk* ch = mtmd_input_chunks_get(chunks, c);
            if (mtmd_input_chunk_get_type(ch) == MTMD_INPUT_CHUNK_TYPE_IMAGE) ichunk = ch;
        }
        if (err.empty() && !ichunk) err = "no image chunk";
        if (err.empty() && mtmd_encode_chunk(ctx, ichunk) != 0) err = "the vision encoder failed";
        if (err.empty()) {
            const mtmd_image_tokens* it = mtmd_input_chunk_get_tokens_image(ichunk);
            const int n = (int) mtmd_input_chunk_get_n_tokens(ichunk);
            // the grid from the decoder positions (nx/ny getters are deprecated): x and y of the last token
            const mtmd_decoder_pos last = mtmd_image_tokens_get_decoder_pos(it, 0, (size_t) n - 1);
            const int nx = (int) last.x + 1, ny = (int) last.y + 1;
            if (nx * ny != n) err = "the image grid is not rectangular (" + std::to_string(n) + " tokens)";
            const float* embd = mtmd_get_output_embd(ctx);
            FILE* f = err.empty() ? std::fopen(out.c_str(), "wb") : nullptr;
            if (err.empty() && !f) err = "cannot write " + out;
            if (f) {
                const int32_t hdr[5] = {0x31455653, n, nx, ny, n_embd};
                const bool ok = std::fwrite(hdr, sizeof hdr, 1, f) == 1 &&
                                std::fwrite(embd, sizeof(float) * (size_t) n_embd, (size_t) n, f) == (size_t) n;
                std::fclose(f);
                if (!ok) err = "short write to " + out;
                const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (ok) std::printf("OK %d %d %d %.0f\n", n, nx, ny, ms);
            }
        }
        if (!err.empty()) std::printf("ERR %s\n", err.c_str());
        std::fflush(stdout);
        mtmd_input_chunks_free(chunks);
        mtmd_bitmap_free(bw.bitmap);
        if (bw.video_ctx) mtmd_helper_video_free(bw.video_ctx);
    }
    mtmd_free(ctx);
    llama_model_free(text);
    llama_backend_free();
    return 0;
}
