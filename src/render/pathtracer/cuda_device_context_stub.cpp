#include "render/pathtracer/cuda_device_context.h"

#include <stdexcept>

namespace renderer {

std::optional<CudaDeviceContext> CudaDeviceContext::try_create(
    int,
    std::string* reason) {
    if (reason) {
        *reason = "renderer was built without CUDA support";
    }
    return std::nullopt;
}

CudaDeviceContext CudaDeviceContext::create(int) {
    throw std::runtime_error(
        "CUDA device selection failed: renderer was built without CUDA support");
}

void CudaDeviceContext::activate() const {
    throw std::runtime_error("renderer was built without CUDA support");
}

bool CudaDeviceContext::is_current() const {
    return false;
}

}  // namespace renderer
