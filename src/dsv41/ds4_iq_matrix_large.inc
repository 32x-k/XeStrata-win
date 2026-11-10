// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// An independent register-output GEMM using XeStrata's IQ1 decoder and WMMA.
// STAGE 0: larger tile; 1: K64 metadata lookahead; 2: double-buffered X.
// Keep the original ascending K16 accumulation and both half roundings.
constexpr size_t kLargeRowGroups = 8;

template <int T>
uint32_t large_packet(const uint8_t* row, int kb, int group) {
    const uint8_t* b = row + size_t(kb >> 3) * Ty<T>::B;
    const int ib = kb & 7;
    uint32_t index, sign, scale, base;
    if constexpr (T == kIQ1_S) {
        const uint32_t high = uint32_t(b[34 + 2 * ib]) | uint32_t(b[35 + 2 * ib]) << 8;
        index = b[2 + 4 * ib + group] | (((high >> (3 * group)) & 7u) << 8);
        sign = high >> 15; scale = (high >> 12) & 7u;
        base = uint32_t(b[0]) | uint32_t(b[1]) << 8;
    } else {
        const uint32_t high = b[32 + 2 * ib + group / 2];
        index = b[4 * ib + group] | (((high >> (4 * (group & 1))) & 7u) << 8);
        sign = (high >> (3 + 4 * (group & 1))) & 1u;
        const uint32_t scales = uint32_t(b[48 + 2 * (ib >> 1)]) | uint32_t(b[49 + 2 * (ib >> 1)]) << 8;
        scale = (scales >> (6 * (ib & 1) + 3 * (group / 2))) & 7u;
        base = iq1m_d(b);
    }
    return index | (sign << 11) | (scale << 12) | (base << 16);
}

[[maybe_unused]] sycl::vec<uint16_t, 8> large_expand(uint32_t packet, const uint32_t* tab) {
    const uint32_t grid = tab[packet & 2047u];
    const float d = sycl_detail::f16_value(uint16_t(packet >> 16)) * 0.125f;
    const float scale = matrix_half_value(matrix_half_bits(d * float(2 * int((packet >> 12) & 7u) + 1)));
    sycl::vec<uint16_t, 8> result;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int code = 8 * int((grid >> ((j & 3) * 8 + (j >> 2) * 4)) & 15u) - 7 - ((packet & 2048u) ? 2 : 0);
        result[j] = matrix_half_bits(scale * float(code));
    }
    return result;
}

size_t large_register_budget(const sycl::device& device) {
    static core::PerDevice<size_t> cache;
    return cache.get(device, [&]() -> size_t {
#if defined(SYCL_EXT_CODEPLAY_MAX_REGISTERS_PER_WORK_GROUP_QUERY)
    try { return device.get_info<sycl::ext::codeplay::experimental::info::device::max_registers_per_work_group>(); }
    catch (const sycl::exception&) { return 0; }
#else
    (void)device; return 0;
#endif
    });
}

template <int T, bool GU, int STAGE, bool Prefetch = false>
bool large_launch(const SetArgs& a, sycl::queue& s) {
    if constexpr (T != kIQ1_S && T != kIQ1_M) return false;
    else {
        using Kernel = std::conditional_t<Prefetch, PrefetchLargeKernel<T, GU, STAGE>, LargeKernel<T, GU, STAGE>>;
        constexpr size_t RT = kLargeRowGroups, CT = 8, rows = RT * MM, bn = CT * MN, threads = CT * SG;
        constexpr size_t roles = GU ? 2 : 1, pk = STAGE ? 4 * MK : QK;
        constexpr size_t pitch = pk + MK / 2, buffers = STAGE == 2 ? 2 : 1;
        constexpr size_t local = Ty<T>::TAB * sizeof(uint32_t) + rows * sizeof(int) + buffers * rows * pitch * sizeof(uint16_t);
        // The32-row boundary retains p2 below it. Nonzero64/128-token
        // fixtures and representative long-row populations (2026-11-08)
        // passed the unit speed/bit checks at this boundary; resource
        // reports and the compiled-kernel limits select this bounded family.
        constexpr int cutoff = int(2 * MM);
        const int n = GU ? a.n_ff : a.n_embd, k = GU ? a.n_embd : a.n_ff;
        if (a.n_exp <= 0 || a.max_rows <= cutoff || n <= 0 || k <= 0 || k % 256 || n % int(bn) ||
            a.ldx % 8 || a.ldo % 2 || (a.skip_ge > 0 && a.skip_ge <= cutoff + 1)) return false;
        if constexpr (GU) { if (!a.xf && !a.xf16) return false; } else if (!a.xh) return false;
        const void* input = GU ? static_cast<const void*>(a.xf) : static_cast<const void*>(a.xh);
        if ((reinterpret_cast<uintptr_t>(input) & 15u) || (GU && a.xf16 && (reinterpret_cast<uintptr_t>(a.xf16) & 15u))) return false;
        const auto device = s.get_device(); const auto& c = caps(s);
        if (c.group < threads || c.local < local || !device.has(sycl::aspect::fp16) ||
            !xe::mma_shape_is(s, int(MM), int(MN), int(MK), int(SG)) || !xe::amd_wmma(device) ||
            std::find(c.subgroups.begin(), c.subgroups.end(), SG) == c.subgroups.end()) return false;
        // Accumulators, two compact metadata sets, staged X, prefetched X and a conservative
        // expansion/address allowance; absence of the query keeps M1/p2.
        constexpr size_t per_lane = RT * roles * (MM * MN / SG) +
            (STAGE ? 2 * (pk / MK) * roles * 2 : 0) +
            (STAGE == 2 ? rows * pk / threads : 0) +
            (Prefetch && STAGE == 0 ? ((rows * pk / 8 + threads - 1) / threads) * 4 : 0) + 48;
        const size_t budget = large_register_budget(device);
        if (budget < threads * per_lane) return false;
        static core::PerDevice<bool> compiled_caps;
        if (!compiled_caps.get(device, [&] {
            try {
                const auto id = sycl::get_kernel_id<Kernel>();
                const auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(s.get_context(), {device}, {id});
                const auto kernel = bundle.get_kernel(id);
                return kernel.template get_info<sycl::info::kernel_device_specific::work_group_size>(device) >= threads &&
                       kernel.template get_info<sycl::info::kernel_device_specific::private_mem_size>(device) == 0;
            } catch (const sycl::exception&) { return false; }
        })) return false;
        const size_t nc = size_t(n) / bn, nr = (size_t(a.max_rows) + rows - 1) / rows;
        const size_t rb = row_bytes_t<T>(k);
        const int skip = std::max(a.skip_le, cutoff);
        sycl_detail::require_in_order(s);
        if (a.skip_le < cutoff) {
            SetArgs small = a; small.max_rows = cutoff; small.skip_ge = cutoff + 1;
            if (!established_launch<T, GU>(small, s)) return false;
        }
        const MatrixGrid grid = matrix_grid(a, s, rows, nc, skip);
        const size_t groups = grid.groups;
        if (!groups) return true;
        sycl_detail::submit(s, [&](sycl::handler& h) {
            sycl::local_accessor<uint32_t, 1> table_data(sycl::range<1>(Ty<T>::TAB), h);
            sycl::local_accessor<int, 1> sources(sycl::range<1>(rows), h);
            sycl::local_accessor<sycl::half, 1> xs(sycl::range<1>(buffers * rows * pitch), h);
            h.parallel_for<Kernel>(sycl::nd_range<1>(groups * threads, threads),
                large_bound<Prefetch, threads, SG>([a, nc, nr, skip, rb, k, n, table_data, sources, xs, grid]
                (sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(SG), sycl::reqd_work_group_size(threads)]] {
                (void)it; (void)a; (void)nc; (void)nr; (void)skip; (void)rb; (void)k; (void)n;
                (void)table_data; (void)sources; (void)xs; (void)grid;
#if STRATA_AMD_WMMA
                constexpr size_t RT = kLargeRowGroups, CT = 8, rows = RT * MM, bn = CT * MN, threads = CT * SG;
                constexpr size_t roles = GU ? 2 : 1, pk = STAGE ? 4 * MK : QK;
                constexpr size_t pitch = pk + MK / 2, parts = xe::wmma::KN / 8;
                const size_t id = it.get_local_linear_id(), g = it.get_group_linear_id();
                const size_t e = grid.jobs ? size_t(grid.jobs[g / nc][0]) : g / (nc * nr);
                const size_t r0 = grid.jobs ? size_t(grid.jobs[g / nc][1]) : ((g / nc) % nr) * rows;
                const size_t c0 = (g % nc) * bn;
                const int count = a.bounds[e + 1] - a.bounds[e], first = a.bounds[e] - a.row0;
                // Round up to retain zero padding of the last live WMMA block.
                const size_t active_rows = sycl::min(rows, ((size_t(count) - r0 + MM - 1) / MM) * MM);
                if (count <= skip || (a.skip_ge > 0 && count >= a.skip_ge) || r0 >= size_t(count)) return;
                for (size_t j = id; j < size_t(Ty<T>::TAB); j += threads) table_data[j] = table<T>()[j];
                if (id < rows) sources[id] = r0 + id < size_t(count) ?
                    (GU && a.idx ? a.idx[size_t(first) + r0 + id] : first + int(r0 + id)) : -1;
                it.barrier(sycl::access::fence_space::local_space);
                const auto sg = it.get_sub_group();
                const size_t column = sg.get_group_linear_id() * MN + xe::wmma::lane16(sg);
                const auto* grids = table_data.template get_multi_ptr<sycl::access::decorated::no>().get();
                const auto xp = xs.template get_multi_ptr<sycl::access::decorated::no>();
                const uint8_t* w[roles];
#pragma unroll
                for (size_t role = 0; role < roles; ++role)
                    w[role] = a.wbase + e * a.stride + (GU ? (role ? a.tr[1] : a.tr[0]) : a.tr[2]) + (c0 + column) * rb;
                xe::wmma::C accum[RT][roles];
#pragma unroll
                for (size_t r = 0; r < RT; ++r)
#pragma unroll
                    for (size_t role = 0; role < roles; ++role) xe::wmma::fill(sg, accum[r][role], 0.f);
                uint32_t packets[pk / MK][roles][parts] = {};
                if constexpr (STAGE > 0) {
#pragma unroll
                    for (size_t step = 0; step < pk / MK; ++step)
#pragma unroll
                        for (size_t role = 0; role < roles; ++role)
#pragma unroll
                            for (size_t part = 0; part < parts; ++part)
                                packets[step][role][part] = large_packet<T>(w[role], int(step * MK / QK),
                                    int((step * MK % QK + xe::wmma::half_of(sg) * xe::wmma::KOFF) / 8 + part));
                }
                auto read_x = [&](size_t j, size_t base) {
                    const int src = sources[j / pk]; uint16_t value = 0;
                    if (src >= 0) {
                        const size_t at = size_t(src) * a.ldx + base + j % pk;
                        value = GU ? (a.xf16 ? a.xf16[at] : sycl_detail::f16_bits(a.xf[at])) : a.xh[at];
                    }
                    return sycl::bit_cast<sycl::half>(value);
                };
                static_assert(pk%8==0 && pitch%8==0);
                using XPack=sycl::vec<sycl::half,8>;
                auto read_pack=[&](size_t j,size_t base) {
                    sycl::vec<uint16_t,8> bits(0);const int src=sources[j/(pk/8)];
                    if(src>=0) {
                        const size_t at=size_t(src)*a.ldx+base+(j%(pk/8))*8;
                        const uint16_t* aligned=static_cast<const uint16_t*>(__builtin_assume_aligned((GU ? a.xf16 : a.xh)+at,16));
                        bits.load(0,sycl::address_space_cast<sycl::access::address_space::global_space,sycl::access::decorated::no>(aligned));
                    }
                    return bits.template as<XPack>();
                };
                auto store_pack=[&](size_t j,const XPack& value) {
                    const size_t dst=(j/(pk/8))*pitch+(j%(pk/8))*8;
                    value.store(0,xp+std::ptrdiff_t(dst));
                };
                constexpr size_t packs=(rows*pk/8+threads-1)/threads;
                if constexpr(Prefetch && STAGE==0) if(!GU || a.xf16) {
#pragma unroll
                    for(size_t p=0;p<packs;++p){size_t j=id+p*threads;if(j<active_rows*pk/8)store_pack(j,read_pack(j,0));}
                    it.barrier(sycl::access::fence_space::local_space);
                }
                if constexpr (STAGE == 2) {
                    for (size_t j = id; j < active_rows * pk; j += threads) xs[(j / pk) * pitch + j % pk] = read_x(j, 0);
                    it.barrier(sycl::access::fence_space::local_space);
                }
                for (size_t base = 0; base < size_t(k); base += pk) {
                    XPack future[packs];
                    if constexpr(Prefetch && STAGE==0) if((!GU || a.xf16) && base+pk<size_t(k)) {
#pragma unroll
                        for(size_t p=0;p<packs;++p){size_t j=id+p*threads;if(j<active_rows*pk/8)future[p]=read_pack(j,base+pk);}
                    }
                    uint32_t next[pk / MK][roles][parts] = {};
                    if constexpr (STAGE > 0) {
                        if (base + pk < size_t(k)) {
#pragma unroll
                            for (size_t step = 0; step < pk / MK; ++step)
#pragma unroll
                                for (size_t role = 0; role < roles; ++role)
#pragma unroll
                                    for (size_t part = 0; part < parts; ++part)
                                        next[step][role][part] = large_packet<T>(w[role], int((base + pk + step * MK) / QK),
                                            int((step * MK % QK + xe::wmma::half_of(sg) * xe::wmma::KOFF) / 8 + part));
                        }
                    }
                    const size_t slot = STAGE == 2 ? (base / pk) % 2 : 0;
                    if constexpr (STAGE == 2) {
                        // The other LDS face is free after the previous
                        // iteration's barrier. Stage directly into it: X need
                        // not remain live beside all GU accumulators. Current
                        // MADs read only this iteration's face; the end barrier
                        // makes the next face visible before its first read.
                        if (base + pk < size_t(k))
                            for (size_t j = id; j < active_rows * pk; j += threads)
                                xs[(1 - slot) * rows * pitch + (j / pk) * pitch + j % pk] = read_x(j, base + pk);
                    } else {
                        if constexpr(Prefetch && STAGE==0) {
                            if(!GU || a.xf16) {
                            } else {
                                for(size_t j=id;j<active_rows*pk;j+=threads)xs[(j/pk)*pitch+j%pk]=read_x(j,base);
                                it.barrier(sycl::access::fence_space::local_space);
                            }
                        } else {
                            for(size_t j=id;j<active_rows*pk;j+=threads)xs[(j/pk)*pitch+j%pk]=read_x(j,base);
                            it.barrier(sycl::access::fence_space::local_space);
                        }
                    }
#pragma unroll
                    for (size_t step = 0; step < pk / MK; ++step) {
                        xe::wmma::Bc<sycl::half> bv[roles];
#pragma unroll
                        for (size_t role = 0; role < roles; ++role)
#pragma unroll
                            for (size_t part = 0; part < parts; ++part) {
                                sycl::vec<uint16_t, 8> bits;
                                if constexpr (STAGE > 0) bits = large_expand(packets[step][role][part], grids);
                                else bits = register_eight<T>(w[role], int((base + step * MK) / QK),
                                    int((step * MK % QK + xe::wmma::half_of(sg) * xe::wmma::KOFF) / 8 + part), grids);
#pragma unroll
                                for (int j = 0; j < 8; ++j) bv[role].v[part * 8 + size_t(j)] = sycl::bit_cast<_Float16>(bits[j]);
                            }
#pragma unroll
                        for (size_t r = 0; r < RT; ++r) {
                            if (r0 + r * MM < size_t(count)) {
                                xe::wmma::A<sycl::half> av;
                                xe::wmma::load(sg, av, xp + std::ptrdiff_t((slot * rows + r * MM) * pitch + step * MK), pitch);
#pragma unroll
                                for (size_t role = 0; role < roles; ++role) xe::wmma::mad(sg, accum[r][role], av, bv[role], accum[r][role]);
                            }
                        }
                    }
                    if constexpr (STAGE > 0) {
#pragma unroll
                        for (size_t step = 0; step < pk / MK; ++step)
#pragma unroll
                            for (size_t role = 0; role < roles; ++role)
#pragma unroll
                                for (size_t part = 0; part < parts; ++part) packets[step][role][part] = next[step][role][part];
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    if constexpr(Prefetch && STAGE==0) if((!GU || a.xf16) && base+pk<size_t(k)) {
#pragma unroll
                        for(size_t p=0;p<packs;++p){size_t j=id+p*threads;if(j<active_rows*pk/8)store_pack(j,future[p]);}
                        it.barrier(sycl::access::fence_space::local_space);
                    }
                }
#pragma unroll
                for (size_t r = 0; r < RT; ++r)
#pragma unroll
                    for (int j = 0; j < 8; ++j) {
                        const size_t row = r0 + r * MM + xe::wmma::row_of(xe::wmma::half_of(sg), j);
                        if (row >= size_t(count) || row >= size_t(a.max_rows)) continue;
                        const size_t out = (size_t(first) + row) * a.ldo + c0 + column;
                        if constexpr (GU) a.h[out] = sycl_detail::f16_bits(portable::swiglu(accum[r][0].v[j], accum[r][1].v[j], a.limit));
                        else a.y[out] = accum[r][0].v[j];
                    }
#endif
            }));
        });
        return true;
    }
}

template <int T, bool GU>
bool launch(const SetArgs& a, sycl::queue& s) {
    if constexpr (GU && (T == kIQ1_S || T == kIQ1_M)) {
        // One row tile plus one WMMA row block was the conservative crossover
        // measured on R9700 (2026-11-10), IQ1_S/M, K5120/N2304 and K32 staging.
        // It depends on this compiled tile, K and the expert row population;
        // unsupported device resources/layouts retain the existing kernels.
        constexpr int threshold = int((kLargeRowGroups + 1) * MM);
        if (a.xf16 && a.max_rows >= threshold && !(a.skip_ge > 0 && a.skip_ge <= threshold)) {
            SetArgs high = a; high.skip_le = std::max(a.skip_le, threshold - 1);
            if (large_launch<T, GU, 0, true>(high, s)) {
                if (a.skip_le >= threshold - 1) return true;
                SetArgs low = a;
                low.skip_ge = a.skip_ge > 0 ? std::min(a.skip_ge, threshold) : threshold;
                low.max_rows = std::min(a.max_rows, threshold - 1);
                if (large_launch<T, GU, 0>(low, s)) return true;
                return established_launch<T, GU>(low, s);
            }
        }
    }
    if constexpr (!GU && (T == kIQ1_S || T == kIQ1_M)) {
        // R9700 (2026-11-10), IQ1_S/M, K2304/N5120: the row sweep selected
        // three WMMA row blocks for down, independently of the GU boundary.
        // It depends on K, this compiled tile and the expert row population;
        // unsupported resources or layouts retain the existing path.
        constexpr int threshold = int(3 * MM);
        if (a.xh && a.max_rows >= threshold && !(a.skip_ge > 0 && a.skip_ge <= threshold)) {
            SetArgs high = a; high.skip_le = std::max(a.skip_le, threshold - 1);
            if (large_launch<T, GU, 0, true>(high, s)) {
                if (a.skip_le >= threshold - 1) return true;
                SetArgs low = a;
                low.skip_ge = a.skip_ge > 0 ? std::min(a.skip_ge, threshold) : threshold;
                low.max_rows = std::min(a.max_rows, threshold - 1);
                if (large_launch<T, GU, 0>(low, s)) return true;
                return established_launch<T, GU>(low, s);
            }
        }
    }
    if (large_launch<T, GU, 0>(a, s)) return true;
    return established_launch<T, GU>(a, s);
}
