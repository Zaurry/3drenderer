#pragma once

#include <algorithm>
#include <optional>
#include <span>
#include <string>

namespace renderer {

inline std::optional<int> select_cuda_device_id(
    int requested_device,
    int device_count,
    std::span<const int> compatible_devices,
    bool require_compatible_device,
    std::string* reason = nullptr) {
    if (device_count <= 0) {
        if (reason) {
            *reason = "no CUDA-capable device is available";
        }
        return std::nullopt;
    }
    if (requested_device < -1) {
        if (reason) {
            *reason = "CUDA device id must be -1 (automatic) or non-negative";
        }
        return std::nullopt;
    }
    if (require_compatible_device && compatible_devices.empty()) {
        if (reason) {
            *reason = "the current graphics context has no compatible CUDA device";
        }
        return std::nullopt;
    }
    const int selected = requested_device >= 0
        ? requested_device
        : (compatible_devices.empty() ? 0 : compatible_devices.front());
    if (selected >= device_count) {
        if (reason) {
            *reason = "requested CUDA device " + std::to_string(selected) +
                " is out of range";
        }
        return std::nullopt;
    }
    if (require_compatible_device &&
        std::find(
            compatible_devices.begin(),
            compatible_devices.end(),
            selected) == compatible_devices.end()) {
        if (reason) {
            *reason = "requested CUDA device " + std::to_string(selected) +
                " is not compatible with the current graphics context";
        }
        return std::nullopt;
    }
    if (reason) {
        reason->clear();
    }
    return selected;
}

class CudaDeviceContext {
public:
    CudaDeviceContext() = default;

    static std::optional<CudaDeviceContext> try_create(
        int device_id,
        std::string* reason = nullptr);
    static CudaDeviceContext create(int device_id);

    int device_id() const { return device_id_; }
    bool valid() const { return device_id_ >= 0; }
    void activate() const;
    bool is_current() const;

private:
    explicit CudaDeviceContext(int device_id) : device_id_(device_id) {}

    int device_id_ = -1;
};

}  // namespace renderer
