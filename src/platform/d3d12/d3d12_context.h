#pragma once
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <d3d12.h>
#include <dxgi1_6.h>
#include <wrl/client.h>
#include "render/dxr/dxr_settings.h"
#include <array>
#include <atomic>
#include <filesystem>
#include <memory>
#include <mutex>
#include <map>
#include <span>
#include <vector>

namespace renderer {
template<class T> using ComPtr = Microsoft::WRL::ComPtr<T>;
void check_hr(HRESULT hr, const char* operation);
std::filesystem::path dxr_binary_directory();
std::vector<std::byte> load_dxr_shader(const char* name);

struct D3d12DescriptorPool {
    ComPtr<ID3D12DescriptorHeap> heap;
    UINT stride = 0, next = 1, capacity = 65536;
    std::map<UINT,UINT> free;
    std::mutex mutex;
    UINT allocate(UINT count=1);
    void release(UINT index,UINT count=1);
    D3D12_CPU_DESCRIPTOR_HANDLE cpu(UINT index) const;
    D3D12_GPU_DESCRIPTOR_HANDLE gpu(UINT index) const;
};
struct D3d12Descriptor {
    std::shared_ptr<D3d12DescriptorPool> pool;
    UINT index = 0, count = 1;
    ~D3d12Descriptor() { if(pool && index)pool->release(index,count); }
};
struct D3d12Resource {
    ComPtr<ID3D12Resource> resource;
    D3D12_RESOURCE_STATES state = D3D12_RESOURCE_STATE_COMMON;
    std::shared_ptr<D3d12Descriptor> srv, uav;
    std::uint64_t bytes = 0, allocation_bytes=0;
    std::shared_ptr<std::atomic<std::uint64_t>> accounting;
    ~D3d12Resource(){if(accounting)accounting->fetch_sub(allocation_bytes);}
    UINT srv_index() const {return srv?srv->index:0;}
    UINT uav_index() const {return uav?uav->index:0;}
};
using D3d12ResourcePtr = std::shared_ptr<D3d12Resource>;

// A frame allocation is valid until that command submission completes. Persistent
// shader tables must use upload() instead of this ring allocator.
struct D3d12UploadSlice {
    D3d12ResourcePtr buffer;
    UINT64 offset=0;
    D3D12_GPU_VIRTUAL_ADDRESS address() const {return buffer->resource->GetGPUVirtualAddress()+offset;}
};
struct D3d12ResourcePool;

class DxrStreamline;
class D3d12Context : public std::enable_shared_from_this<D3d12Context> {
public:
    static constexpr UINT timestamp_count=10;
    static std::shared_ptr<D3d12Context> create(bool load_streamline=true);
    ~D3d12Context();
    ID3D12GraphicsCommandList4* begin();
    ID3D12GraphicsCommandList4* commands() const {return list_.Get();}
    std::uint64_t submit();
    std::uint64_t signal_presented();
    void wait(std::uint64_t value);
    void flush();
    bool recording() const {return recording_;}
    bool enhanced_barriers_active() const {return bool(enhanced_list_);}
    std::uint64_t next_fence() const {return fence_value_+1;}
    void retain(const D3d12ResourcePtr& resource);
    void transition(const D3d12ResourcePtr& resource, D3D12_RESOURCE_STATES state);
    void uav_barrier(const D3d12ResourcePtr& resource);
    D3d12ResourcePtr buffer(std::uint64_t bytes, UINT stride = 0,
        D3D12_RESOURCE_FLAGS flags = D3D12_RESOURCE_FLAG_NONE,
        D3D12_RESOURCE_STATES initial = D3D12_RESOURCE_STATE_COMMON);
    D3d12ResourcePtr upload(std::span<const std::byte> bytes);
    D3d12UploadSlice upload_frame(std::span<const std::byte> bytes,UINT64 alignment=256);
    D3d12ResourcePtr upload_buffer(std::span<const std::byte> bytes, UINT stride);
    D3d12ResourcePtr texture(UINT width,UINT height,DXGI_FORMAT format,UINT mips=1,
        bool storage=true);
    D3d12ResourcePtr upload_texture(UINT width,UINT height,DXGI_FORMAT format,
        UINT bytes_per_pixel,std::span<const std::byte> bytes);
    D3d12ResourcePtr readback_buffer(std::uint64_t bytes);
    std::shared_ptr<D3d12Descriptor> descriptor(UINT count=1);
    UINT frame_index() const {return frame_;}
    void timestamp(UINT stage);
    void set_timing_tag(std::uint64_t tag) {frames_[frame_].timing_tag=tag;}
    std::uint64_t completed_timing_tag() const {return completed_timing_tag_;}
    std::array<float,timestamp_count> timings() const {return timings_;}
    const DxrDeviceCapabilities& capabilities() const {return capabilities_;}
    ID3D12Device5* device() const {return device_.Get();}
    ID3D12CommandQueue* queue() const {return queue_.Get();}
    IDXGIFactory6* factory() const {return factory_.Get();}
    const std::shared_ptr<D3d12DescriptorPool>& descriptors() const {return descriptors_;}
    ID3D12DescriptorHeap* samplers() const {return samplers_.Get();}
    UINT sampler(bool nearest,unsigned wrap_s,unsigned wrap_t) const;
    std::uint64_t allocated_bytes() const {return allocated_bytes_->load();}
    bool query_video_memory(std::uint64_t& usage,std::uint64_t& budget) const;
    std::uint64_t pooled_bytes() const;
    std::uint64_t resource_creations() const {return resource_creations_;}
    std::uint64_t resource_reuses() const {return resource_reuses_;}
    ComPtr<ID3D12PipelineState> compute_pipeline(const wchar_t* name,const D3D12_COMPUTE_PIPELINE_STATE_DESC&);
    std::uint64_t pipeline_cache_hits() const {return pipeline_cache_hits_;}
    void check_validation() const;
    DxrStreamline* streamline() const {return streamline_.get();}
private:
    explicit D3d12Context(bool load_streamline);
    std::unique_ptr<DxrStreamline> streamline_;
    std::shared_ptr<D3d12ResourcePool> resource_pool_;
    struct UploadPage {D3d12ResourcePtr buffer;std::byte* data=nullptr;UINT64 used=0;};
    struct Frame {
        ComPtr<ID3D12CommandAllocator> allocator;
        std::uint64_t fence=0;
        std::uint64_t timing_tag=0;
        std::vector<D3d12ResourcePtr> retained;
        std::array<bool,timestamp_count> stamps{};
        std::vector<UploadPage> uploads;
    };
    std::array<Frame,3> frames_;
    UINT frame_=2;
    DxrDeviceCapabilities capabilities_;
    ComPtr<IDXGIFactory6> factory_;
    ComPtr<IDXGIAdapter3> memory_adapter_;
    ComPtr<ID3D12Device5> device_;
    ComPtr<ID3D12CommandQueue> queue_;
    ComPtr<ID3D12GraphicsCommandList4> list_;
    ComPtr<ID3D12GraphicsCommandList7> enhanced_list_;
    ComPtr<ID3D12Fence> fence_;
    ComPtr<ID3D12QueryHeap> queries_;
    D3d12ResourcePtr query_readback_;
    std::shared_ptr<D3d12DescriptorPool> descriptors_;
    ComPtr<ID3D12DescriptorHeap> samplers_;
    HANDLE event_=nullptr;
    std::uint64_t fence_value_=0, frequency_=1;
    std::shared_ptr<std::atomic<std::uint64_t>> allocated_bytes_=std::make_shared<std::atomic<std::uint64_t>>(0);
    bool recording_=false;
    std::array<float,timestamp_count> timings_{};
    std::uint64_t completed_timing_tag_=0;
    std::uint64_t resource_creations_=0,resource_reuses_=0,pipeline_cache_hits_=0;
    std::vector<std::byte> pipeline_library_data_;
    ComPtr<ID3D12PipelineLibrary> pipeline_library_;
    std::filesystem::path pipeline_cache_path_;
    bool pipeline_cache_dirty_=false;
    D3d12ResourcePtr pooled_resource(const D3D12_RESOURCE_DESC&,UINT stride,D3D12_RESOURCE_STATES);
    void save_pipeline_cache() noexcept;
};

class DxrFrameResource {
public:
    std::shared_ptr<D3d12Context> context;
    D3d12ResourcePtr color;
};
} // namespace renderer
#endif
