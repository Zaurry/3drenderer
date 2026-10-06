#pragma once
#include "platform/d3d12/d3d12_context.h"
#include "render/display_settings.h"
#include "render/interactive/render_frame_output.h"
struct ImDrawData;
namespace renderer {
class D3d12Presenter {
public:
    D3d12Presenter(HWND window,int width,int height);
    ~D3d12Presenter();
    void new_ui_frame();
    void present(const DxrTextureHandle&,const DisplaySettings&,int width,int height,ImDrawData*);
    const std::shared_ptr<D3d12Context>& context() const {return context_;}
private:
    std::shared_ptr<D3d12Context> context_;
    ComPtr<IDXGISwapChain3> swapchain_;
    ComPtr<ID3D12DescriptorHeap> rtv_heap_;
    ComPtr<ID3D12RootSignature> root_;
    ComPtr<ID3D12PipelineState> pipeline_;
    std::array<D3d12ResourcePtr,3> backbuffers_;
    std::array<std::uint64_t,3> fences_{};
    UINT rtv_stride_=0,flags_=0;
    int width_=0,height_=0;
    bool imgui_=false;
    void resize(int width,int height);
};
} // namespace renderer
