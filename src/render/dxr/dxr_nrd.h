#pragma once
#include "platform/d3d12/d3d12_context.h"
#include "render/dxr/dxr_gpu.h"
#include <memory>
namespace renderer {
struct DxrNrdInputs {
    D3d12ResourcePtr motion,normal_roughness,view_z,diffuse,specular;
};
class DxrNrd {
public:
    explicit DxrNrd(std::shared_ptr<D3d12Context>);
    ~DxrNrd();
    void resize(UINT width,UINT height,bool validation=false);
    void dispatch(const DxrFrameConstants&,const DxrNrdInputs&,float delta_seconds);
    const D3d12ResourcePtr& diffuse() const;
    const D3d12ResourcePtr& specular() const;
    const D3d12ResourcePtr& validation() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace renderer
