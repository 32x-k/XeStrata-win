// SPDX-FileCopyrightText: 2026 MistVVK and the XeStrata contributors
// SPDX-License-Identifier: LGPL-3.0-or-later
// include/strata/image/safetensors.hpp - a safetensors file, read: its header (an 8-byte length, then a JSON object
// of tensors: {"name": {"dtype": "BF16", "shape": [..], "data_offsets": [begin, end]}, "__metadata__": {..}}) and
// its tensors' bytes.  The header is parsed for these fields only; anything else in it is skipped.
#pragma once

#include <cstdint>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace strata::image {

struct SafeTensor {
    std::string dtype;
    std::vector<int64_t> shape;
    uint64_t begin = 0, end = 0;   // within the data section
    int64_t elements() const {
        int64_t n = 1;
        for (int64_t d : shape) n *= d;
        return n;
    }
};

class SafeTensors {
public:
    explicit SafeTensors(const std::string& path) : path_(path), f_(path, std::ios::binary) {
        if (!f_) throw std::runtime_error("cannot open " + path);
        uint64_t n = 0;
        f_.read(reinterpret_cast<char*>(&n), 8);
        if (!f_ || n == 0 || n > (64u << 20)) throw std::runtime_error(path + ": not a safetensors file");
        std::string h(n, '\0');
        f_.read(h.data(), (std::streamsize) n);
        if (!f_) throw std::runtime_error(path + ": a short header");
        base_ = 8 + n;
        f_.seekg(0, std::ios::end);
        size_ = (uint64_t) f_.tellg();
        parse(h);
    }

    const SafeTensor* find(const std::string& name) const {
        auto it = t_.find(name);
        return it == t_.end() ? nullptr : &it->second;
    }
    const std::map<std::string, SafeTensor>& tensors() const { return t_; }
    /// The header's `__metadata__`: its string values (a HunyuanImage 3.0 file names its model's config there).  Values that are
    /// not strings are skipped.
    const std::map<std::string, std::string>& metadata() const { return meta_; }
    /// A metadata value, or "".
    std::string meta(const std::string& key) const {
        auto it = meta_.find(key);
        return it == meta_.end() ? std::string() : it->second;
    }
    /// Where the data section starts in the file (a tensor's bytes are at data_base() + begin .. + end).
    uint64_t data_base() const { return base_; }

    /// The tensor's values as FP32 (BF16, F16 and F32 files).
    std::vector<float> f32(const std::string& name) {
        const SafeTensor* t = find(name);
        if (t == nullptr) throw std::runtime_error(path_ + ": no tensor " + name);
        // #30: the offsets are checked against the file before anything is allocated from them
        if (t->end < t->begin || base_ > size_ || t->end > size_ - base_)
            throw std::runtime_error(path_ + ": " + name + ": data_offsets outside the file");
        std::vector<char> raw(t->end - t->begin);
        f_.seekg((std::streamoff) (base_ + t->begin));
        f_.read(raw.data(), (std::streamsize) raw.size());
        if (!f_) throw std::runtime_error(path_ + ": cannot read " + name);
        const auto n = (size_t) t->elements();
        std::vector<float> out(n);
        if (t->dtype == "F32" && raw.size() == n * 4) {
            std::memcpy(out.data(), raw.data(), n * 4);
        } else if (t->dtype == "BF16" && raw.size() == n * 2) {
            for (size_t i = 0; i < n; ++i) {
                uint32_t u = (uint32_t) (uint8_t) raw[2 * i] | ((uint32_t) (uint8_t) raw[2 * i + 1] << 8);
                u <<= 16;
                std::memcpy(&out[i], &u, 4);
            }
        } else if (t->dtype == "F16" && raw.size() == n * 2) {
            for (size_t i = 0; i < n; ++i) {
                const uint16_t h = (uint16_t) ((uint8_t) raw[2 * i] | ((uint8_t) raw[2 * i + 1] << 8));
                const uint32_t s = (uint32_t) (h & 0x8000u) << 16, e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
                uint32_t u;
                if (e == 0) {
                    float f = (float) m * 5.9604644775390625e-8f;   // subnormal: m x 2^-24
                    std::memcpy(&u, &f, 4);
                    u |= s;
                } else if (e == 31) {
                    u = s | 0x7F800000u | (m << 13);
                } else {
                    u = s | ((e + 112) << 23) | (m << 13);
                }
                std::memcpy(&out[i], &u, 4);
            }
        } else {
            throw std::runtime_error(path_ + ": " + name + ": dtype " + t->dtype);
        }
        return out;
    }

private:
    // #30: the character at i, or '\0' past the end (a header cut short reads as one that ends there)
    static char at(const std::string& h, size_t i) { return i < h.size() ? h[i] : '\0'; }
    void ws(const std::string& h, size_t& i) {
        while (i < h.size() && (h[i] == ' ' || h[i] == '\n' || h[i] == '\r' || h[i] == '\t')) ++i;
    }
    void expect(const std::string& h, size_t& i, char c) {
        ws(h, i);
        if (i >= h.size() || h[i] != c) throw std::runtime_error(path_ + ": header: expected '" + std::string(1, c) + "'");
        ++i;
    }
    std::string str(const std::string& h, size_t& i) {
        expect(h, i, '"');
        std::string s;
        while (i < h.size() && h[i] != '"') {
            if (h[i] == '\\' && i + 1 < h.size()) ++i;
            s += h[i++];
        }
        if (i >= h.size()) throw std::runtime_error(path_ + ": header: an unterminated string");   // #30
        ++i;
        return s;
    }
    int64_t num(const std::string& h, size_t& i) {
        ws(h, i);
        size_t j = i;
        while (j < h.size() && ((h[j] >= '0' && h[j] <= '9') || h[j] == '-')) ++j;
        if (j == i) throw std::runtime_error(path_ + ": header: expected a number");
        const int64_t v = std::stoll(h.substr(i, j - i));
        i = j;
        return v;
    }
    std::vector<int64_t> nums(const std::string& h, size_t& i) {
        std::vector<int64_t> v;
        expect(h, i, '[');
        ws(h, i);
        if (at(h, i) == ']') return ++i, v;
        for (;;) {
            v.push_back(num(h, i));
            ws(h, i);
            if (at(h, i) == ',') { ++i; continue; }
            expect(h, i, ']');
            return v;
        }
    }
    void skip(const std::string& h, size_t& i) {   // any JSON value
        ws(h, i);
        if (at(h, i) == '"') { str(h, i); return; }
        if (at(h, i) == '{' || at(h, i) == '[') {
            const char open = h[i], close = open == '{' ? '}' : ']';
            int depth = 0;
            bool in_str = false;
            for (; i < h.size(); ++i) {
                if (in_str) {
                    if (h[i] == '\\' && i + 1 < h.size()) ++i;
                    else if (h[i] == '"') in_str = false;
                } else if (h[i] == '"') in_str = true;
                else if (h[i] == open) ++depth;
                else if (h[i] == close && --depth == 0) { ++i; return; }
            }
            return;
        }
        while (i < h.size() && h[i] != ',' && h[i] != '}' && h[i] != ']') ++i;
    }
    void parse(const std::string& h) {
        size_t i = 0;
        expect(h, i, '{');
        for (;;) {
            ws(h, i);
            if (at(h, i) == '}') return;
            const std::string name = str(h, i);
            expect(h, i, ':');
            if (name == "__metadata__") {
                expect(h, i, '{');
                for (;;) {
                    ws(h, i);
                    if (at(h, i) == '}') { ++i; break; }
                    const std::string key = str(h, i);
                    expect(h, i, ':');
                    ws(h, i);
                    if (at(h, i) == '"') meta_[key] = str(h, i);
                    else skip(h, i);
                    ws(h, i);
                    if (at(h, i) == ',') ++i;
                }
            } else {
                SafeTensor t;
                expect(h, i, '{');
                for (;;) {
                    ws(h, i);
                    if (at(h, i) == '}') { ++i; break; }
                    const std::string key = str(h, i);
                    expect(h, i, ':');
                    if (key == "dtype") t.dtype = str(h, i);
                    else if (key == "shape") {
                        t.shape = nums(h, i);
                        for (int64_t d : t.shape)
                            if (d < 0) throw std::runtime_error(path_ + ": header: a negative dimension");
                    }
                    else if (key == "data_offsets") {
                        const auto o = nums(h, i);
                        if (o.size() != 2) throw std::runtime_error(path_ + ": header: data_offsets");
                        t.begin = (uint64_t) o[0];
                        t.end = (uint64_t) o[1];
                    } else skip(h, i);
                    ws(h, i);
                    if (at(h, i) == ',') ++i;
                }
                t_[name] = t;
            }
            ws(h, i);
            if (at(h, i) == ',') ++i;
        }
    }

    std::string path_;
    std::ifstream f_;
    uint64_t base_ = 0, size_ = 0;
    std::map<std::string, SafeTensor> t_;
    std::map<std::string, std::string> meta_;
};

}  // namespace strata::image
