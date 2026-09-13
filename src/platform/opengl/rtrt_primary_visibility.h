#pragma once

#include "render/realtime/primary_visibility.h"

namespace renderer {

std::shared_ptr<RealtimePrimaryVisibility> make_opengl_primary_visibility(CudaDeviceContext);

} // namespace renderer
