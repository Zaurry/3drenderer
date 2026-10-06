#include "platform/d3d12/d3d12_context.h"
#include <iostream>
int main() {
    const auto c=renderer::query_dxr_capabilities();
    std::cout<<"adapter="<<c.adapter<<" available="<<c.available<<" dxr_tier="<<c.raytracing_tier
        <<" shader_model="<<std::hex<<c.shader_model<<std::dec<<" ser_supported="<<c.ser_supported
        <<" ser_reorders="<<c.ser_reorders<<" omm="<<c.omm_supported<<" enhanced_barriers="<<c.enhanced_barriers
        <<" dedicated_bytes="<<c.dedicated_bytes<<" reason="<<c.reason<<'\n';
    return c.available?0:1;
}
