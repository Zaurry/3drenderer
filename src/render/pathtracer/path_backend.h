#pragma once

#include "render/render_settings.h"
#include "render/renderer.h"

#include <string>

namespace renderer {

ExecutionBackend resolve_path_backend(PathBackend requested);
const char* execution_backend_name(ExecutionBackend backend);
PathBackend parse_path_backend(const std::string& value);

}  // namespace renderer
