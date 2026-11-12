// SPDX-FileCopyrightText: 2026 Peasant Smith and the Project Maya contributors
// SPDX-FileCopyrightText: 2026 Niko1221 and the Strata contributors
// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later AND MIT
#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace strata::ds4 {

struct VisionGeo {
    int layers = 0, dim = 1024, heads = 16, inter = 2816, patch = 14, merge = 3, out = 5120;
    float theta = 10000.0f, eps = 1e-6f;
    int max_tokens = 1024;   // image span: merged rows * (merged columns + newline) + start/end
};

class VisionTower {
public:
    /// Return the shape, and, when dst is not null, the tensor converted to BF16 bits.
    using Reader = std::function<bool(const std::string& name, std::vector<int64_t>& shape, void* dst)>;
    VisionTower();
    ~VisionTower();
    VisionTower(const VisionTower&) = delete;
    VisionTower& operator=(const VisionTower&) = delete;
    /// Validate all shapes before reading weights. No device memory or queue is kept by load.
    bool load(const VisionGeo& g, int device, const Reader& read, std::string& err);
    void release();
    bool loaded() const;
    int device() const;
    int max_patches() const;
    size_t host_bytes() const;
    /// Peak workspace estimate for the last image, including Gemm's conversion buffers.
    size_t workspace_bytes() const;
    /// RGB rows have patch-aligned sides. Return merged patch rows, without image span markers.
    /// All copies and kernels finish and the device workspace is released before this returns.
    bool encode(const uint8_t* rgb, int width, int height, std::vector<uint16_t>& rows, int& n_h, int& n_w,
                std::string& err);

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace strata::ds4
