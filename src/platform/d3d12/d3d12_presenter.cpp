#include "platform/d3d12/d3d12_presenter.h"
#include "render/dxr/dxr_streamline.h"
#include <imgui.h>
#include <imgui_impl_dx12.h>
#include <bit>
#include <cmath>

namespace renderer {
D3d12Presenter::D3d12Presenter(HWND window,int width,int height):context_(D3d12Context::create()) {
    BOOL tearing=FALSE;
    if(SUCCEEDED(context_->factory()->CheckFeatureSupport(DXGI_FEATURE_PRESENT_ALLOW_TEARING,&tearing,sizeof(tearing))) && tearing)flags_=DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING;
    DXGI_SWAP_CHAIN_DESC1 desc{};desc.Width=std::max(1,width);desc.Height=std::max(1,height);desc.Format=DXGI_FORMAT_R8G8B8A8_UNORM;
    desc.SampleDesc.Count=1;desc.BufferUsage=DXGI_USAGE_RENDER_TARGET_OUTPUT;desc.BufferCount=3;desc.SwapEffect=DXGI_SWAP_EFFECT_FLIP_DISCARD;desc.Flags=flags_;
    ComPtr<IDXGISwapChain1> swap;check_hr(context_->factory()->CreateSwapChainForHwnd(context_->queue(),window,&desc,nullptr,nullptr,&swap),"Create DXR swapchain");
    if(context_->streamline())context_->streamline()->upgrade_swapchain(swap);
    check_hr(swap.As(&swapchain_),"Query swapchain3");context_->factory()->MakeWindowAssociation(window,DXGI_MWA_NO_ALT_ENTER);
    D3D12_DESCRIPTOR_HEAP_DESC heap{};heap.Type=D3D12_DESCRIPTOR_HEAP_TYPE_RTV;heap.NumDescriptors=3;
    check_hr(context_->device()->CreateDescriptorHeap(&heap,IID_PPV_ARGS(&rtv_heap_)),"Create swapchain RTV heap");rtv_stride_=context_->device()->GetDescriptorHandleIncrementSize(heap.Type);
    D3D12_ROOT_PARAMETER parameter{};parameter.ParameterType=D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;parameter.Constants={0,0,4};
    D3D12_ROOT_SIGNATURE_DESC root_desc{};root_desc.NumParameters=1;root_desc.pParameters=&parameter;
    root_desc.Flags=D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED|D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED;
    ComPtr<ID3DBlob> blob,error;check_hr(D3D12SerializeRootSignature(&root_desc,D3D_ROOT_SIGNATURE_VERSION_1,&blob,&error),"Serialize compositor root");
    check_hr(context_->device()->CreateRootSignature(0,blob->GetBufferPointer(),blob->GetBufferSize(),IID_PPV_ARGS(&root_)),"Create compositor root");
    const auto vs=load_dxr_shader("present_vs.dxil"),ps=load_dxr_shader("present_ps.dxil");D3D12_GRAPHICS_PIPELINE_STATE_DESC pipeline{};
    pipeline.pRootSignature=root_.Get();pipeline.VS={vs.data(),vs.size()};pipeline.PS={ps.data(),ps.size()};pipeline.SampleMask=UINT_MAX;
    pipeline.RasterizerState.FillMode=D3D12_FILL_MODE_SOLID;pipeline.RasterizerState.CullMode=D3D12_CULL_MODE_NONE;pipeline.RasterizerState.DepthClipEnable=TRUE;
    pipeline.BlendState.RenderTarget[0].RenderTargetWriteMask=D3D12_COLOR_WRITE_ENABLE_ALL;pipeline.PrimitiveTopologyType=D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE;
    pipeline.NumRenderTargets=1;pipeline.RTVFormats[0]=desc.Format;pipeline.SampleDesc.Count=1;
    check_hr(context_->device()->CreateGraphicsPipelineState(&pipeline,IID_PPV_ARGS(&pipeline_)),"Create compositor pipeline");resize(width,height);
    ImGui_ImplDX12_InitInfo info{};info.Device=context_->device();info.CommandQueue=context_->queue();info.NumFramesInFlight=3;info.RTVFormat=desc.Format;
    info.SrvDescriptorHeap=context_->descriptors()->heap.Get();info.UserData=context_.get();
    info.SrvDescriptorAllocFn=[](ImGui_ImplDX12_InitInfo* i,D3D12_CPU_DESCRIPTOR_HANDLE* cpu,D3D12_GPU_DESCRIPTOR_HANDLE* gpu) {
        auto& pool=static_cast<D3d12Context*>(i->UserData)->descriptors();UINT index=pool->allocate();*cpu=pool->cpu(index);*gpu=pool->gpu(index);
    };
    info.SrvDescriptorFreeFn=[](ImGui_ImplDX12_InitInfo* i,D3D12_CPU_DESCRIPTOR_HANDLE cpu,D3D12_GPU_DESCRIPTOR_HANDLE) {
        auto& context=*static_cast<D3d12Context*>(i->UserData);const auto& pool=context.descriptors();
        auto owner=std::make_shared<D3d12Resource>();owner->srv=std::make_shared<D3d12Descriptor>();owner->srv->pool=pool;
        owner->srv->index=UINT((cpu.ptr-pool->cpu(0).ptr)/pool->stride);
        if(context.recording())context.retain(owner);else context.flush();
    };
    if(!ImGui_ImplDX12_Init(&info))throw std::runtime_error("ImGui DX12 initialization failed");imgui_=true;
}
D3d12Presenter::~D3d12Presenter(){try{context_->flush();}catch(...){}if(imgui_)ImGui_ImplDX12_Shutdown();}
void D3d12Presenter::new_ui_frame(){ImGui_ImplDX12_NewFrame();}
void D3d12Presenter::resize(int width,int height) {
    width=std::max(1,width);height=std::max(1,height);if(width_==width && height_==height)return;
    context_->flush();for(auto& b:backbuffers_)b.reset();fences_.fill(0);
    if(width_)check_hr(swapchain_->ResizeBuffers(3,width,height,DXGI_FORMAT_R8G8B8A8_UNORM,flags_),"Resize DXR swapchain");
    width_=width;height_=height;
    for(UINT i=0;i<3;++i){auto b=std::make_shared<D3d12Resource>();b->state=D3D12_RESOURCE_STATE_PRESENT;check_hr(swapchain_->GetBuffer(i,IID_PPV_ARGS(&b->resource)),"Get swapchain buffer");
        auto rtv=rtv_heap_->GetCPUDescriptorHandleForHeapStart();rtv.ptr+=SIZE_T(i)*rtv_stride_;context_->device()->CreateRenderTargetView(b->resource.Get(),nullptr,rtv);backbuffers_[i]=std::move(b);}
}
void D3d12Presenter::present(const DxrTextureHandle& frame,const DisplaySettings& display,int width,int height,ImDrawData* ui) {
    if(!frame.resource || frame.resource->context!=context_)throw std::runtime_error("DXR frame belongs to a different presentation device");
    if(width<=0 || height<=0){context_->submit();return;}
    resize(width,height);if(!context_->recording())context_->begin();
    const UINT index=swapchain_->GetCurrentBackBufferIndex();context_->wait(fences_[index]);auto* command=context_->commands();
    context_->transition(backbuffers_[index],D3D12_RESOURCE_STATE_RENDER_TARGET);context_->transition(frame.resource->color,D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);
    auto rtv=rtv_heap_->GetCPUDescriptorHandleForHeapStart();rtv.ptr+=SIZE_T(index)*rtv_stride_;
    command->OMSetRenderTargets(1,&rtv,FALSE,nullptr);D3D12_VIEWPORT viewport{0,0,float(width),float(height),0,1};D3D12_RECT scissor{0,0,width,height};
    command->RSSetViewports(1,&viewport);command->RSSetScissorRects(1,&scissor);command->SetGraphicsRootSignature(root_.Get());command->SetPipelineState(pipeline_.Get());
    const UINT constants[]={frame.resource->color->srv_index(),std::bit_cast<UINT>(std::isfinite(display.exposure_ev)?display.exposure_ev:0.f),UINT(display.tone_mapper),0};
    command->SetGraphicsRoot32BitConstants(0,4,constants,0);command->IASetPrimitiveTopology(D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST);command->DrawInstanced(3,1,0,0);
    context_->timestamp(8);
    ImGui_ImplDX12_RenderDrawData(ui,command);context_->transition(backbuffers_[index],D3D12_RESOURCE_STATE_PRESENT);context_->timestamp(9);fences_[index]=context_->submit();
    check_hr(swapchain_->Present(0,flags_?DXGI_PRESENT_ALLOW_TEARING:0),"Present DXR frame");
    fences_[index]=context_->signal_presented();
}
} // namespace renderer
