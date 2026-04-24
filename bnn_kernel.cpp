#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <arm_neon.h>
#include <cstdint>
#include <cstring>
#include <vector>

// Fast popcount over packed bits.
static inline uint32_t popcount_xor(const uint8_t* __restrict a,
                                    const uint8_t* __restrict w,
                                    int K_bytes) {
    uint32_t acc = 0;
    int b = 0;
    for (; b <= K_bytes - 16; b += 16) {
        uint8x16_t va = vld1q_u8(a + b);
        uint8x16_t vw = vld1q_u8(w + b);
        uint8x16_t xr = veorq_u8(va, vw);
        uint8x16_t pc = vcntq_u8(xr);
        uint16x8_t p16 = vpaddlq_u8(pc);
        uint32x4_t p32 = vpaddlq_u16(p16);
        acc += vaddvq_u32(p32);
    }
    for (; b < K_bytes; ++b) {
        acc += static_cast<uint32_t>(__builtin_popcount(a[b] ^ w[b]));
    }
    return acc;
}

// Same popcount, but ignore padded bits.
static inline uint32_t popcount_xor_masked(const uint8_t* __restrict a,
                                           const uint8_t* __restrict w,
                                           const uint8_t* __restrict m,
                                           int K_bytes) {
    uint32_t acc = 0;
    int b = 0;

    for (; b <= K_bytes - 16; b += 16) {
        uint8x16_t va = vld1q_u8(a + b);
        uint8x16_t vw = vld1q_u8(w + b);
        uint8x16_t vm = vld1q_u8(m + b);
        uint8x16_t xr = veorq_u8(va, vw);
        uint8x16_t masked = vandq_u8(xr, vm);
        uint8x16_t pc = vcntq_u8(masked);
        uint16x8_t p16 = vpaddlq_u8(pc);
        uint32x4_t p32 = vpaddlq_u16(p16);
        acc += vaddvq_u32(p32);
    }
    for (; b < K_bytes; ++b) {
        acc += static_cast<uint32_t>(__builtin_popcount((a[b] ^ w[b]) & m[b]));
    }
    return acc;
}

// Plain packed binary GEMM.
torch::Tensor bgemm_neon(torch::Tensor packed_in,
                         torch::Tensor packed_w,
                         int64_t K) {
    TORCH_CHECK(packed_in.is_contiguous(), "packed_in must be contiguous");
    TORCH_CHECK(packed_w.is_contiguous(),  "packed_w must be contiguous");
    TORCH_CHECK(packed_in.dtype() == torch::kUInt8, "packed_in must be uint8");
    TORCH_CHECK(packed_w.dtype()  == torch::kUInt8, "packed_w must be uint8");
    TORCH_CHECK(packed_in.dim() == 2, "packed_in must be 2D");
    TORCH_CHECK(packed_w.dim()  == 2, "packed_w must be 2D");
    TORCH_CHECK(packed_in.size(1) == packed_w.size(1),
                "packed_in and packed_w must have matching K_bytes");

    const int64_t N       = packed_in.size(0);
    const int64_t M       = packed_w.size(0);
    const int64_t K_bytes = packed_in.size(1);

    auto output = torch::empty({N, M}, torch::TensorOptions().dtype(torch::kFloat32));

    const uint8_t* a_base = packed_in.data_ptr<uint8_t>();
    const uint8_t* w_base = packed_w.data_ptr<uint8_t>();
    float*         o_base = output.data_ptr<float>();

    const int Kb = static_cast<int>(K_bytes);
    const int Ki = static_cast<int>(K);

    at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const uint8_t* a = a_base + i * Kb;
            float*         o = o_base + i * M;
            for (int64_t j = 0; j < M; ++j) {
                const uint8_t* w = w_base + j * Kb;
                uint32_t pc = popcount_xor(a, w, Kb);
                o[j] = static_cast<float>(Ki - 2 * static_cast<int>(pc));
            }
        }
    });
    return output;
}

// Masked version for padded windows.
torch::Tensor bgemm_neon_masked(torch::Tensor packed_in,
                                torch::Tensor packed_mask,
                                torch::Tensor packed_w,
                                torch::Tensor k_valid) {
    TORCH_CHECK(packed_in.is_contiguous(),   "packed_in must be contiguous");
    TORCH_CHECK(packed_mask.is_contiguous(), "packed_mask must be contiguous");
    TORCH_CHECK(packed_w.is_contiguous(),    "packed_w must be contiguous");
    TORCH_CHECK(k_valid.is_contiguous(),     "k_valid must be contiguous");
    TORCH_CHECK(packed_in.dtype()   == torch::kUInt8, "packed_in must be uint8");
    TORCH_CHECK(packed_mask.dtype() == torch::kUInt8, "packed_mask must be uint8");
    TORCH_CHECK(packed_w.dtype()    == torch::kUInt8, "packed_w must be uint8");
    TORCH_CHECK(k_valid.dtype()     == torch::kInt32, "k_valid must be int32");
    TORCH_CHECK(packed_in.sizes() == packed_mask.sizes(),
                "packed_in and packed_mask must have the same shape");
    TORCH_CHECK(packed_in.size(1) == packed_w.size(1),
                "packed_in and packed_w must have matching K_bytes");
    TORCH_CHECK(k_valid.size(0) == packed_in.size(0),
                "k_valid must have length N");

    const int64_t N       = packed_in.size(0);
    const int64_t M       = packed_w.size(0);
    const int64_t K_bytes = packed_in.size(1);

    auto output = torch::empty({N, M}, torch::TensorOptions().dtype(torch::kFloat32));

    const uint8_t* a_base = packed_in.data_ptr<uint8_t>();
    const uint8_t* m_base = packed_mask.data_ptr<uint8_t>();
    const uint8_t* w_base = packed_w.data_ptr<uint8_t>();
    const int32_t* kv     = k_valid.data_ptr<int32_t>();
    float*         o_base = output.data_ptr<float>();

    const int Kb = static_cast<int>(K_bytes);

    at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const uint8_t* a = a_base + i * Kb;
            const uint8_t* m = m_base + i * Kb;
            float*         o = o_base + i * M;
            const int      kv_i = kv[i];
            for (int64_t j = 0; j < M; ++j) {
                const uint8_t* w = w_base + j * Kb;
                uint32_t pc = popcount_xor_masked(a, w, m, Kb);
                o[j] = static_cast<float>(kv_i - 2 * static_cast<int>(pc));
            }
        }
    });
    return output;
}

// Match numpy.packbits: MSB first.
torch::Tensor pack_bits_msb(torch::Tensor bits) {
    TORCH_CHECK(bits.is_contiguous(),           "bits must be contiguous");
    TORCH_CHECK(bits.dtype() == torch::kUInt8,  "bits must be uint8");
    TORCH_CHECK(bits.dim() == 2,                "bits must be 2-D [N, K]");
    const int64_t N = bits.size(0);
    const int64_t K = bits.size(1);
    TORCH_CHECK(K % 8 == 0, "K must be a multiple of 8");
    const int64_t Kb = K / 8;

    auto out = torch::empty({N, Kb}, torch::TensorOptions().dtype(torch::kUInt8));
    const uint8_t* in_base  = bits.data_ptr<uint8_t>();
    uint8_t*       out_base = out.data_ptr<uint8_t>();

    at::parallel_for(0, N, 1, [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
            const uint8_t* in  = in_base + i * K;
            uint8_t*       o   = out_base + i * Kb;
            int64_t b = 0;

            for (; b + 16 <= K; b += 16) {
                uint8x16_t v = vld1q_u8(in + b);
                uint8x16_t ones = vdupq_n_u8(1);
                uint8x16_t bits01 = vandq_u8(
                    vcgtq_u8(v, vdupq_n_u8(0)), ones);
                alignas(16) static const uint8_t WEIGHTS[16] = {
                    128, 64, 32, 16, 8, 4, 2, 1,
                    128, 64, 32, 16, 8, 4, 2, 1,
                };
                uint8x16_t w = vld1q_u8(WEIGHTS);
                uint8x16_t weighted = vmulq_u8(bits01, w);
                uint8x8_t lo = vget_low_u8(weighted);
                uint8x8_t hi = vget_high_u8(weighted);
                o[b / 8 + 0] = vaddv_u8(lo);
                o[b / 8 + 1] = vaddv_u8(hi);
            }
            for (; b < K; b += 8) {
                uint8_t byte = 0;
                for (int k = 0; k < 8; ++k) {
                    if (in[b + k]) byte |= static_cast<uint8_t>(1u << (7 - k));
                }
                o[b / 8] = byte;
            }
        }
    });
    return out;
}

// Pack eight signs into one byte.
static inline uint8_t sign_pack_8(const float* p) {
    uint8_t b = 0;
    b |= (p[0] >= 0.0f) ? 0x80 : 0;
    b |= (p[1] >= 0.0f) ? 0x40 : 0;
    b |= (p[2] >= 0.0f) ? 0x20 : 0;
    b |= (p[3] >= 0.0f) ? 0x10 : 0;
    b |= (p[4] >= 0.0f) ? 0x08 : 0;
    b |= (p[5] >= 0.0f) ? 0x04 : 0;
    b |= (p[6] >= 0.0f) ? 0x02 : 0;
    b |= (p[7] >= 0.0f) ? 0x01 : 0;
    return b;
}

// Fuse im2col, sign, and packing.
torch::Tensor im2col_sign_pack(torch::Tensor x,
                               int64_t kH, int64_t kW,
                               int64_t pH, int64_t pW,
                               int64_t sH, int64_t sW) {
    TORCH_CHECK(x.is_contiguous(),            "x must be contiguous");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "x must be float32");
    TORCH_CHECK(x.dim() == 4,                 "x must be 4-D [N, C, H, W]");

    const int64_t N = x.size(0);
    const int64_t C = x.size(1);
    const int64_t H = x.size(2);
    const int64_t W = x.size(3);
    const int64_t K = C * kH * kW;
    TORCH_CHECK(C % 8 == 0, "C must be a multiple of 8 in channel-inner layout");
    const int64_t Cb = C / 8;
    const int64_t Kb = K / 8;
    const int64_t H_out = (H + 2 * pH - kH) / sH + 1;
    const int64_t W_out = (W + 2 * pW - kW) / sW + 1;
    const int64_t L = H_out * W_out;

    auto out = torch::zeros({N * L, Kb}, torch::TensorOptions().dtype(torch::kUInt8));
    const float* x_base = x.data_ptr<float>();
    uint8_t*     o_base = out.data_ptr<uint8_t>();

    const int64_t HW = H * W;
    const int64_t CHW = C * HW;

    at::parallel_for(0, N * L, 16, [&](int64_t begin, int64_t end) {
        std::vector<float> scratch(C);
        for (int64_t idx = begin; idx < end; ++idx) {
            const int64_t n    = idx / L;
            const int64_t loc  = idx - n * L;
            const int64_t hout = loc / W_out;
            const int64_t wout = loc - hout * W_out;
            uint8_t* o = o_base + idx * Kb;
            const float* x_n = x_base + n * CHW;

            for (int64_t ky = 0; ky < kH; ++ky) {
                const int64_t ih = hout * sH - pH + ky;
                for (int64_t kx = 0; kx < kW; ++kx) {
                    const int64_t iw = wout * sW - pW + kx;
                    uint8_t* tap_out = o + ((ky * kW) + kx) * Cb;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        for (int64_t c = 0; c < C; ++c) {
                            scratch[c] = x_n[c * HW + ih * W + iw];
                        }
                        for (int64_t bc = 0; bc < Cb; ++bc) {
                            tap_out[bc] = sign_pack_8(scratch.data() + bc * 8);
                        }
                    }
                }
            }
        }
    });
    return out;
}

// Build mask once per spatial location.
std::vector<torch::Tensor> build_mask(int64_t N, int64_t C,
                                      int64_t H, int64_t W,
                                      int64_t kH, int64_t kW,
                                      int64_t pH, int64_t pW,
                                      int64_t sH, int64_t sW) {
    const int64_t K = C * kH * kW;
    TORCH_CHECK(C % 8 == 0, "C must be a multiple of 8 in channel-inner layout");
    const int64_t Cb = C / 8;
    const int64_t Kb = K / 8;
    const int64_t H_out = (H + 2 * pH - kH) / sH + 1;
    const int64_t W_out = (W + 2 * pW - kW) / sW + 1;
    const int64_t L = H_out * W_out;

    auto mask    = torch::zeros({N * L, Kb},
                    torch::TensorOptions().dtype(torch::kUInt8));
    auto k_valid = torch::empty({N * L},
                    torch::TensorOptions().dtype(torch::kInt32));

    uint8_t* m_base = mask.data_ptr<uint8_t>();
    int32_t* k_base = k_valid.data_ptr<int32_t>();

    at::parallel_for(0, L, 16, [&](int64_t begin, int64_t end) {
        for (int64_t loc = begin; loc < end; ++loc) {
            const int64_t hout = loc / W_out;
            const int64_t wout = loc - hout * W_out;
            int valid_count_kk = 0;
            uint8_t* row0 = m_base + loc * Kb;
            for (int64_t ky = 0; ky < kH; ++ky) {
                const int64_t ih = hout * sH - pH + ky;
                for (int64_t kx = 0; kx < kW; ++kx) {
                    const int64_t iw = wout * sW - pW + kx;
                    uint8_t* tap_out = row0 + ((ky * kW) + kx) * Cb;
                    if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                        std::memset(tap_out, 0xFF, Cb);
                        ++valid_count_kk;
                    } else {
                        std::memset(tap_out, 0x00, Cb);
                    }
                }
            }
            const int32_t kv = static_cast<int32_t>(valid_count_kk * C);
            k_base[loc] = kv;
            for (int64_t n = 1; n < N; ++n) {
                uint8_t* dst = m_base + (n * L + loc) * Kb;
                std::memcpy(dst, row0, Kb);
                k_base[n * L + loc] = kv;
            }
        }
    });
    return {mask, k_valid};
}

// Reorder weights to match im2col_sign_pack.
torch::Tensor repack_weight_khwc(torch::Tensor packed_chwm, int64_t C,
                                 int64_t kH, int64_t kW) {
    TORCH_CHECK(packed_chwm.is_contiguous(),           "must be contiguous");
    TORCH_CHECK(packed_chwm.dtype() == torch::kUInt8,  "must be uint8");
    TORCH_CHECK(packed_chwm.dim() == 2,                "must be 2-D [M, K/8]");
    const int64_t M = packed_chwm.size(0);
    const int64_t K = C * kH * kW;
    const int64_t Kb = K / 8;
    TORCH_CHECK(packed_chwm.size(1) == Kb, "Kb mismatch");
    TORCH_CHECK(C % 8 == 0, "C must be a multiple of 8");

    auto out = torch::empty({M, Kb}, torch::TensorOptions().dtype(torch::kUInt8));
    const uint8_t* src = packed_chwm.data_ptr<uint8_t>();
    uint8_t*       dst = out.data_ptr<uint8_t>();

    auto read_bit = [&](const uint8_t* row, int64_t bit_idx) -> int {
        const int64_t b = bit_idx >> 3;
        const int sh = 7 - static_cast<int>(bit_idx & 7);
        return (row[b] >> sh) & 1;
    };
    auto set_bit = [&](uint8_t* row, int64_t bit_idx, int v) {
        const int64_t b = bit_idx >> 3;
        const int sh = 7 - static_cast<int>(bit_idx & 7);
        if (v) row[b] |= static_cast<uint8_t>(1u << sh);
    };

    at::parallel_for(0, M, 1, [&](int64_t begin, int64_t end) {
        for (int64_t m = begin; m < end; ++m) {
            const uint8_t* s = src + m * Kb;
            uint8_t*       d = dst + m * Kb;
            std::memset(d, 0, Kb);
            for (int64_t c = 0; c < C; ++c) {
                for (int64_t ky = 0; ky < kH; ++ky) {
                    for (int64_t kx = 0; kx < kW; ++kx) {
                        const int64_t src_bit = c * (kH * kW) + ky * kW + kx;
                        const int v = read_bit(s, src_bit);
                        const int64_t dst_bit = (ky * kW + kx) * C + c;
                        set_bit(d, dst_bit, v);
                    }
                }
            }
        }
    });
    return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("bgemm_neon",         &bgemm_neon,         "NEON binary GEMM (no mask)");
    m.def("bgemm_neon_masked",  &bgemm_neon_masked,  "NEON binary GEMM with validity mask");
    m.def("pack_bits_msb",      &pack_bits_msb,      "MSB-first bit packing");
    m.def("im2col_sign_pack",   &im2col_sign_pack,
          "Fused im2col + sign + bit-pack (channel-inner layout)");
    m.def("build_mask",         &build_mask,
          "Build packed validity mask + k_valid (channel-inner layout)");
    m.def("repack_weight_khwc", &repack_weight_khwc,
          "Repack weights to (kH, kW, C) channel-inner layout");
}
