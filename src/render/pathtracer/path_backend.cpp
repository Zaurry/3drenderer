#include "render/pathtracer/path_backend.h"

#include "render/pathtracer/cuda_pathtracer.h"

#include <stdexcept>
#include <string>

namespace renderer {

ExecutionBackend resolve_path_backend(PathBackend requested) {
    if (requested == PathBackend::Cpu) {
        return ExecutionBackend::Cpu;
    }

    std::string reason;
    if (cuda_path_backend_available(&reason)) {
        return ExecutionBackend::Cuda;
    }
    if (requested == PathBackend::Cuda) {
        throw std::runtime_error("CUDA path backend is unavailable: " + reason);
    }
    return ExecutionBackend::Cpu;
}

const char* execution_backend_name(ExecutionBackend backend) {
    return backend == ExecutionBackend::Cuda ? "cuda" : "cpu";
}

PathBackend parse_path_backend(const std::string& value) {
    if (value == "auto") {
        return PathBackend::Auto;
    }
    if (value == "cpu") {
        return PathBackend::Cpu;
    }
    if (value == "cuda") {
        return PathBackend::Cuda;
    }
    throw std::invalid_argument("unknown path backend: " + value);
}

}  // namespace renderer
