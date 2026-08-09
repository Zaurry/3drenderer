#pragma once

#include <filesystem>
#include <string_view>

namespace renderer {

enum class AtomicWriteFailurePoint {
    None,
    BeforeReplace,
};

// Writes a complete replacement in the destination directory and publishes it
// with one platform atomic-replace operation. The existing file is never
// removed before the replacement succeeds.
void write_file_atomically(
    const std::filesystem::path& path,
    std::string_view contents);

// Deterministic one-shot fault injection for document persistence tests.
void set_atomic_write_failure_for_testing(AtomicWriteFailurePoint point);

}  // namespace renderer
