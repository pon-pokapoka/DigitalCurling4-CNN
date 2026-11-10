// Copyright (c) 2026 Takehiro Yoshioka
// SPDX-License-Identifier: MIT

// Minimal SHA-256 (FIPS 180-4), used to identify the model and win table files
// a client loaded. Not for security purposes.

#ifndef SHA256_HPP
#define SHA256_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>

namespace digitalcurling::client {

class Sha256 {
public:
    Sha256() { Reset(); }

    void Reset()
    {
        state_ = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                  0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
        buffer_len_ = 0;
        total_len_ = 0;
    }

    void Update(void const* data, std::size_t len)
    {
        auto const* p = static_cast<std::uint8_t const*>(data);
        total_len_ += len;
        while (len > 0) {
            std::size_t const n = std::min(len, buffer_.size() - buffer_len_);
            for (std::size_t i = 0; i < n; ++i) buffer_[buffer_len_ + i] = p[i];
            buffer_len_ += n;
            p += n;
            len -= n;
            if (buffer_len_ == buffer_.size()) {
                Compress(buffer_.data());
                buffer_len_ = 0;
            }
        }
    }

    // Lowercase hex digest. The object must be Reset() before reuse.
    std::string HexDigest()
    {
        std::uint64_t const bit_len = total_len_ * 8;
        std::uint8_t const pad = 0x80;
        Update(&pad, 1);
        std::uint8_t const zero = 0;
        while (buffer_len_ != 56) Update(&zero, 1);
        std::uint8_t len_be[8];
        for (int i = 0; i < 8; ++i) len_be[i] = static_cast<std::uint8_t>(bit_len >> (56 - 8 * i));
        Update(len_be, 8);

        std::string hex;
        char buf[9];
        for (auto word : state_) {
            std::snprintf(buf, sizeof(buf), "%08x", word);
            hex += buf;
        }
        return hex;
    }

private:
    static std::uint32_t Rotr(std::uint32_t x, int n) { return (x >> n) | (x << (32 - n)); }

    void Compress(std::uint8_t const* block)
    {
        static constexpr std::uint32_t k[64] = {
            0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u, 0xab1c5ed5u,
            0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu, 0x9bdc06a7u, 0xc19bf174u,
            0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu, 0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau,
            0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u, 0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u,
            0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu, 0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u,
            0xa2bfe8a1u, 0xa81a664bu, 0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u,
            0x19a4c116u, 0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
            0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u, 0xc67178f2u};

        std::uint32_t w[64];
        for (int i = 0; i < 16; ++i) {
            w[i] = (std::uint32_t(block[4 * i]) << 24) | (std::uint32_t(block[4 * i + 1]) << 16)
                 | (std::uint32_t(block[4 * i + 2]) << 8) | std::uint32_t(block[4 * i + 3]);
        }
        for (int i = 16; i < 64; ++i) {
            std::uint32_t const s0 = Rotr(w[i - 15], 7) ^ Rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
            std::uint32_t const s1 = Rotr(w[i - 2], 17) ^ Rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16] + s0 + w[i - 7] + s1;
        }

        std::uint32_t a = state_[0], b = state_[1], c = state_[2], d = state_[3];
        std::uint32_t e = state_[4], f = state_[5], g = state_[6], h = state_[7];
        for (int i = 0; i < 64; ++i) {
            std::uint32_t const S1 = Rotr(e, 6) ^ Rotr(e, 11) ^ Rotr(e, 25);
            std::uint32_t const ch = (e & f) ^ (~e & g);
            std::uint32_t const t1 = h + S1 + ch + k[i] + w[i];
            std::uint32_t const S0 = Rotr(a, 2) ^ Rotr(a, 13) ^ Rotr(a, 22);
            std::uint32_t const maj = (a & b) ^ (a & c) ^ (b & c);
            std::uint32_t const t2 = S0 + maj;
            h = g; g = f; f = e; e = d + t1;
            d = c; c = b; b = a; a = t1 + t2;
        }
        state_[0] += a; state_[1] += b; state_[2] += c; state_[3] += d;
        state_[4] += e; state_[5] += f; state_[6] += g; state_[7] += h;
    }

    std::array<std::uint32_t, 8> state_;
    std::array<std::uint8_t, 64> buffer_;
    std::size_t buffer_len_;
    std::uint64_t total_len_;
};

// SHA-256 of a file's contents as lowercase hex. Throws if the file can't be read.
inline std::string FileSha256(std::string const& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open " + path + " to hash it");
    Sha256 sha;
    char buf[1 << 16];
    while (in.read(buf, sizeof(buf)) || in.gcount() > 0) {
        sha.Update(buf, static_cast<std::size_t>(in.gcount()));
    }
    if (in.bad()) throw std::runtime_error("cannot read " + path + " to hash it");
    return sha.HexDigest();
}

} // namespace digitalcurling::client

#endif // SHA256_HPP
