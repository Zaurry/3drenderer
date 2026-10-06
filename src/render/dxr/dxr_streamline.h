#pragma once
#include "platform/d3d12/d3d12_context.h"
#include "render/dxr/dxr_gpu.h"
namespace renderer {
struct DxrDlssInputs {
    D3d12ResourcePtr color,depth,motion,normal_roughness,diffuse_albedo,specular_albedo,specular_distance,specular_motion;
};
class DxrStreamline {
public:
    DxrStreamline();
    ~DxrStreamline();
    void shutdown();
    void release_resources(D3d12Context&);
    DxrStreamline(const DxrStreamline&)=delete;
    void bind_device(ID3D12Device*,DxrDeviceCapabilities&);
    void upgrade_factory(ComPtr<IDXGIFactory6>&);
    void upgrade_swapchain(ComPtr<IDXGISwapChain1>&);
    void create_queue(ID3D12Device*,const D3D12_COMMAND_QUEUE_DESC&,ComPtr<ID3D12CommandQueue>&);
    DxrReconstruction select(DxrReconstruction requested) const;
    std::pair<UINT,UINT> render_size(DxrReconstruction,UINT width,UINT height);
    bool evaluate(D3d12Context&,DxrReconstruction,const DxrFrameConstants&,const DxrDlssInputs&,const D3d12ResourcePtr& output);
    const std::string& reason() const;
    std::uint64_t allocated_bytes() const;
    const std::vector<DxrRuntimeModule>& runtime_modules() const;
    bool runtime_pinned() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
