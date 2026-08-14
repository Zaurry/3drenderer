#pragma once

#include <cmath>

// Single source of truth for MIS weighting, compiled for both host C++ and
// CUDA device code. Edge behavior is canonical: an invalid (non-positive or
// non-finite) first pdf yields 0, an invalid second pdf yields 1.
//
// pbr.cpp and cuda_pathtracer.cu previously carried divergent copies; keep
// them unified through this header.

#if defined(__CUDACC__) || defined(__NVCC__)
#define RENDERER_HOST_DEVICE __host__ __device__
#else
#define RENDERER_HOST_DEVICE
#endif

namespace renderer {

inline RENDERER_HOST_DEVICE float power_heuristic(
    float first_pdf,
    float second_pdf) {
#if defined(__CUDACC__) || defined(__NVCC__)
    if (!(first_pdf > 0.0f) || !isfinite(first_pdf)) {
#else
    if (!(first_pdf > 0.0f) || !std::isfinite(first_pdf)) {
#endif
        return 0.0f;
    }
#if defined(__CUDACC__) || defined(__NVCC__)
    if (!(second_pdf > 0.0f) || !isfinite(second_pdf)) {
#else
    if (!(second_pdf > 0.0f) || !std::isfinite(second_pdf)) {
#endif
        return 1.0f;
    }
    const float first_squared = first_pdf * first_pdf;
    const float second_squared = second_pdf * second_pdf;
    return first_squared / (first_squared + second_squared);
}

}  // namespace renderer

#undef RENDERER_HOST_DEVICE
