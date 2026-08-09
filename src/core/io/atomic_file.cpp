#include "core/io/atomic_file.h"

#include <atomic>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>

#ifdef _WIN32
#define NOMINMAX
#include <windows.h>
#else
#include <unistd.h>
#endif

namespace renderer {
namespace {

std::atomic<AtomicWriteFailurePoint> injected_failure{
    AtomicWriteFailurePoint::None};

std::uint64_t process_id() {
#ifdef _WIN32
    return static_cast<std::uint64_t>(GetCurrentProcessId());
#else
    return static_cast<std::uint64_t>(getpid());
#endif
}

std::filesystem::path temporary_path_for(
    const std::filesystem::path& destination) {
    static std::atomic<std::uint64_t> sequence{0};
    return destination.parent_path() /
        (destination.filename().string() + ".tmp." +
         std::to_string(process_id()) + "." +
         std::to_string(sequence.fetch_add(1, std::memory_order_relaxed)));
}

void remove_temporary(const std::filesystem::path& path) noexcept {
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
}

}  // namespace

void set_atomic_write_failure_for_testing(AtomicWriteFailurePoint point) {
    injected_failure.store(point, std::memory_order_release);
}

void write_file_atomically(
    const std::filesystem::path& path,
    std::string_view contents) {
    if (path.empty()) {
        throw std::invalid_argument("atomic file destination is empty");
    }

    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::error_code directory_error;
        std::filesystem::create_directories(parent, directory_error);
        if (directory_error) {
            throw std::runtime_error(
                "failed to create destination directory: " +
                directory_error.message());
        }
    }

    const std::filesystem::path temporary = temporary_path_for(path);
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "failed to open temporary file for writing: " + temporary.string());
    }
    output.write(contents.data(), static_cast<std::streamsize>(contents.size()));
    output.flush();
    output.close();
    if (!output) {
        remove_temporary(temporary);
        throw std::runtime_error(
            "failed to write temporary file: " + temporary.string());
    }

    if (injected_failure.exchange(
            AtomicWriteFailurePoint::None,
            std::memory_order_acq_rel) ==
        AtomicWriteFailurePoint::BeforeReplace) {
        remove_temporary(temporary);
        throw std::runtime_error(
            "injected atomic write failure before replacement");
    }

#ifdef _WIN32
    if (!MoveFileExW(
            temporary.c_str(),
            path.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        const unsigned long windows_error = GetLastError();
        remove_temporary(temporary);
        throw std::runtime_error(
            "failed to atomically replace file, Windows error " +
            std::to_string(windows_error));
    }
#else
    std::error_code replace_error;
    std::filesystem::rename(temporary, path, replace_error);
    if (replace_error) {
        remove_temporary(temporary);
        throw std::runtime_error(
            "failed to atomically replace file: " + replace_error.message());
    }
#endif
}

}  // namespace renderer
