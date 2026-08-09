#include "render/pathtracer/cuda_device_context.h"

#include <cuda_runtime.h>

#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace renderer {
namespace {

__global__ void cuda_probe_kernel(int* result) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        *result = 0x3d;
    }
}

struct ProbeResult {
    bool available = false;
    std::string reason;
};

ProbeResult probe_device(int device_id) {
    int count = 0;
    cudaError_t result = cudaGetDeviceCount(&count);
    if (result != cudaSuccess) {
        const std::string reason = cudaGetErrorString(result);
        cudaGetLastError();
        return {false, reason};
    }
    if (device_id < 0 || device_id >= count) {
        return {
            false,
            "CUDA device index " + std::to_string(device_id) +
                " is out of range (device count " + std::to_string(count) + ")"};
    }
    result = cudaSetDevice(device_id);
    if (result != cudaSuccess) {
        return {false, std::string("cudaSetDevice failed: ") + cudaGetErrorString(result)};
    }
    int* device_result = nullptr;
    result = cudaMalloc(&device_result, sizeof(int));
    if (result == cudaSuccess) {
        cuda_probe_kernel<<<1, 1>>>(device_result);
        result = cudaGetLastError();
    }
    if (result == cudaSuccess) {
        result = cudaDeviceSynchronize();
    }
    int host_result = 0;
    if (result == cudaSuccess) {
        result = cudaMemcpy(
            &host_result,
            device_result,
            sizeof(host_result),
            cudaMemcpyDeviceToHost);
    }
    if (device_result) {
        const cudaError_t free_result = cudaFree(device_result);
        if (result == cudaSuccess) {
            result = free_result;
        }
    }
    if (result != cudaSuccess) {
        const std::string reason =
            std::string("CUDA probe kernel failed: ") + cudaGetErrorString(result);
        cudaGetLastError();
        return {false, reason};
    }
    if (host_result != 0x3d) {
        return {false, "CUDA probe kernel returned invalid data"};
    }
    return {true, {}};
}

ProbeResult cached_probe(int device_id) {
    static std::mutex mutex;
    static std::unordered_map<int, ProbeResult> results;
    std::lock_guard lock(mutex);
    const auto found = results.find(device_id);
    if (found != results.end()) {
        if (found->second.available) {
            cudaSetDevice(device_id);
        }
        return found->second;
    }
    ProbeResult result = probe_device(device_id);
    results.emplace(device_id, result);
    return result;
}

}  // namespace

std::optional<CudaDeviceContext> CudaDeviceContext::try_create(
    int device_id,
    std::string* reason) {
    const ProbeResult probe = cached_probe(device_id);
    if (reason) {
        *reason = probe.reason;
    }
    if (!probe.available) {
        return std::nullopt;
    }
    return CudaDeviceContext(device_id);
}

CudaDeviceContext CudaDeviceContext::create(int device_id) {
    std::string reason;
    auto context = try_create(device_id, &reason);
    if (!context) {
        throw std::runtime_error("CUDA device selection failed: " + reason);
    }
    return *context;
}

void CudaDeviceContext::activate() const {
    if (!valid()) {
        throw std::logic_error("cannot activate an invalid CUDA device context");
    }
    const cudaError_t result = cudaSetDevice(device_id_);
    if (result != cudaSuccess) {
        throw std::runtime_error(
            std::string("cudaSetDevice failed: ") + cudaGetErrorString(result));
    }
}

bool CudaDeviceContext::is_current() const {
    int current = -1;
    return valid() && cudaGetDevice(&current) == cudaSuccess &&
        current == device_id_;
}

}  // namespace renderer
