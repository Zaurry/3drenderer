#include "platform/d3d12/d3d12_context.h"
#include "render/dxr/dxr_streamline.h"
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <deque>
#include <tuple>
#include <d3d12sdklayers.h>

extern "C" {
__declspec(dllexport) extern const UINT D3D12SDKVersion=619;
__declspec(dllexport) extern const char* D3D12SDKPath=".\\D3D12\\";
}

namespace renderer {
struct D3d12ResourcePool {
    using Key=std::tuple<UINT,UINT64,UINT,UINT,UINT,UINT,UINT,bool>;
    struct Entry {Key key;std::unique_ptr<D3d12Resource> resource;};
    std::deque<Entry> entries;
    std::mutex mutex;
    UINT64 bytes=0;
    static constexpr UINT64 budget=128ull*1024*1024;
    void recycle(const Key& key,D3d12Resource* value) {
        std::unique_ptr<D3d12Resource> resource(value);
        if(!resource->resource || resource->allocation_bytes>budget)return;
        std::lock_guard lock(mutex);
        while(bytes+resource->allocation_bytes>budget && !entries.empty()){bytes-=entries.front().resource->allocation_bytes;entries.pop_front();}
        bytes+=resource->allocation_bytes;entries.push_back({key,std::move(resource)});
    }
};
void check_hr(HRESULT hr,const char* operation) {
    if(SUCCEEDED(hr))return;
    std::ostringstream message;message<<operation<<" failed (HRESULT 0x"<<std::hex<<static_cast<unsigned>(hr)<<")";
    throw std::runtime_error(message.str());
}
std::filesystem::path dxr_binary_directory() {
    std::wstring buffer(32768,L'\0');
    const auto size=GetModuleFileNameW(nullptr,buffer.data(),static_cast<DWORD>(buffer.size()));
    if(!size || size==buffer.size())throw std::runtime_error("cannot locate DXR executable directory");
    buffer.resize(size);return std::filesystem::path(buffer).parent_path();
}
std::vector<std::byte> load_dxr_shader(const char* name) {
    const auto path=dxr_binary_directory()/"shaders"/"dxr"/name;
    std::ifstream input(path,std::ios::binary|std::ios::ate);
    if(!input)throw std::runtime_error("missing DXR shader: "+path.string());
    const auto size=input.tellg();
    if(size<=0)throw std::runtime_error("empty DXR shader: "+path.string());
    std::vector<std::byte> bytes(static_cast<std::size_t>(size));input.seekg(0);
    if(!input.read(reinterpret_cast<char*>(bytes.data()),size))throw std::runtime_error("failed reading DXR shader");
    return bytes;
}
UINT D3d12DescriptorPool::allocate(UINT count) {
    std::lock_guard lock(mutex);
    for(auto it=free.begin();it!=free.end();++it)if(it->second>=count) {
        const UINT first=it->first,remaining=it->second-count;free.erase(it);
        if(remaining)free.emplace(first+count,remaining);return first;
    }
    if(!count || count>capacity-next)throw std::runtime_error("DXR descriptor heap exhausted");
    const UINT first=next;next+=count;return first;
}
void D3d12DescriptorPool::release(UINT index,UINT count) {
    std::lock_guard lock(mutex);auto next_range=free.lower_bound(index);
    if(next_range!=free.begin()) {auto previous=std::prev(next_range);if(previous->first+previous->second==index){index=previous->first;count+=previous->second;free.erase(previous);}}
    if(next_range!=free.end() && index+count==next_range->first){count+=next_range->second;free.erase(next_range);}
    free.emplace(index,count);
}
D3D12_CPU_DESCRIPTOR_HANDLE D3d12DescriptorPool::cpu(UINT index) const {
    auto h=heap->GetCPUDescriptorHandleForHeapStart();h.ptr+=SIZE_T(index)*stride;return h;
}
D3D12_GPU_DESCRIPTOR_HANDLE D3d12DescriptorPool::gpu(UINT index) const {
    auto h=heap->GetGPUDescriptorHandleForHeapStart();h.ptr+=UINT64(index)*stride;return h;
}

namespace {
DxrDeviceCapabilities inspect_device(IDXGIAdapter1* adapter,ID3D12Device5* device) {
    DxrDeviceCapabilities c;DXGI_ADAPTER_DESC1 desc{};adapter->GetDesc1(&desc);
    char name[512]{};WideCharToMultiByte(CP_UTF8,0,desc.Description,-1,name,sizeof(name),nullptr,nullptr);
    c.adapter=name;c.dedicated_bytes=desc.DedicatedVideoMemory;
    std::memcpy(&c.adapter_luid,&desc.AdapterLuid,sizeof(c.adapter_luid));
    D3D12_FEATURE_DATA_D3D12_OPTIONS5 rt{};
    if(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS5,&rt,sizeof(rt))))c.raytracing_tier=rt.RaytracingTier;
    D3D12_FEATURE_DATA_SHADER_MODEL sm{D3D_SHADER_MODEL_6_9};
    if(FAILED(device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL,&sm,sizeof(sm)))) {
        sm.HighestShaderModel=D3D_SHADER_MODEL_6_6;
        device->CheckFeatureSupport(D3D12_FEATURE_SHADER_MODEL,&sm,sizeof(sm));
    }
    c.shader_model=sm.HighestShaderModel;
    D3D12_FEATURE_DATA_D3D12_OPTIONS12 barriers{};
    if(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS12,&barriers,sizeof(barriers))))c.enhanced_barriers=barriers.EnhancedBarriersSupported;
    D3D12_FEATURE_DATA_D3D12_OPTIONS22 ser{};
    if(SUCCEEDED(device->CheckFeatureSupport(D3D12_FEATURE_D3D12_OPTIONS22,&ser,sizeof(ser))))c.ser_reorders=ser.ShaderExecutionReorderingActuallyReorders;
    c.ser_supported=c.shader_model>=D3D_SHADER_MODEL_6_9;
    c.omm_supported=c.raytracing_tier>=D3D12_RAYTRACING_TIER_1_2 && c.ser_supported;
    c.available=c.raytracing_tier>=D3D12_RAYTRACING_TIER_1_1 && c.shader_model>=D3D_SHADER_MODEL_6_6;
    if(!c.available)c.reason="DXR 1.1 and Shader Model 6.6 are required";
    return c;
}
D3D12_HEAP_PROPERTIES heap_properties(D3D12_HEAP_TYPE type) {
    D3D12_HEAP_PROPERTIES p{};p.Type=type;p.CreationNodeMask=p.VisibleNodeMask=1;return p;
}
D3D12_RESOURCE_DESC buffer_desc(std::uint64_t bytes,D3D12_RESOURCE_FLAGS flags) {
    D3D12_RESOURCE_DESC d{};d.Dimension=D3D12_RESOURCE_DIMENSION_BUFFER;d.Width=std::max<std::uint64_t>(bytes,4);
    d.Height=1;d.DepthOrArraySize=1;d.MipLevels=1;d.SampleDesc.Count=1;
    d.Layout=D3D12_TEXTURE_LAYOUT_ROW_MAJOR;d.Flags=flags;return d;
}
struct BarrierScope {D3D12_BARRIER_ACCESS access;D3D12_BARRIER_LAYOUT layout;};
BarrierScope barrier_scope(D3D12_RESOURCE_STATES state) {
    BarrierScope result{D3D12_BARRIER_ACCESS_COMMON,D3D12_BARRIER_LAYOUT_COMMON};
    if(state&D3D12_RESOURCE_STATE_UNORDERED_ACCESS)return {D3D12_BARRIER_ACCESS_UNORDERED_ACCESS,D3D12_BARRIER_LAYOUT_UNORDERED_ACCESS};
    if(state&D3D12_RESOURCE_STATE_RENDER_TARGET)return {D3D12_BARRIER_ACCESS_RENDER_TARGET,D3D12_BARRIER_LAYOUT_RENDER_TARGET};
    if(state&D3D12_RESOURCE_STATE_COPY_DEST)return {D3D12_BARRIER_ACCESS_COPY_DEST,D3D12_BARRIER_LAYOUT_COPY_DEST};
    if(state&D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE)return {D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ|D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE,D3D12_BARRIER_LAYOUT_UNDEFINED};
    if(state&(D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE)){result.access|=D3D12_BARRIER_ACCESS_SHADER_RESOURCE;result.layout=D3D12_BARRIER_LAYOUT_SHADER_RESOURCE;}
    if(state&D3D12_RESOURCE_STATE_COPY_SOURCE){result.access|=D3D12_BARRIER_ACCESS_COPY_SOURCE;result.layout=D3D12_BARRIER_LAYOUT_COPY_SOURCE;}
    if(state&D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER)result.access|=D3D12_BARRIER_ACCESS_VERTEX_BUFFER|D3D12_BARRIER_ACCESS_CONSTANT_BUFFER;
    if(state&D3D12_RESOURCE_STATE_INDEX_BUFFER)result.access|=D3D12_BARRIER_ACCESS_INDEX_BUFFER;
    if(state&D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT)result.access|=D3D12_BARRIER_ACCESS_INDIRECT_ARGUMENT;
    return result;
}
}

std::shared_ptr<D3d12Context> D3d12Context::create(bool load_streamline){return std::shared_ptr<D3d12Context>(new D3d12Context(load_streamline));}
D3d12Context::D3d12Context(bool load_streamline) {
    resource_pool_=std::make_shared<D3d12ResourcePool>();
    static std::once_flag debug_initialization;
    static bool debug_active=false,gpu_validation_active=false;
    std::call_once(debug_initialization,[] {
    if(std::getenv("DXR_VALIDATE") || std::getenv("DXR_GPU_VALIDATION")) {
        ComPtr<ID3D12Debug> debug;
        check_hr(D3D12GetDebugInterface(IID_PPV_ARGS(&debug)),"Enable requested D3D12 debug layer");
        debug->EnableDebugLayer();debug_active=true;
        if(std::getenv("DXR_GPU_VALIDATION")) {
            ComPtr<ID3D12Debug1> gbv;check_hr(debug.As(&gbv),"Enable requested GPU validation");
            gbv->SetEnableGPUBasedValidation(TRUE);gpu_validation_active=true;
        }
        ComPtr<ID3D12DeviceRemovedExtendedDataSettings> dred;
        if(SUCCEEDED(D3D12GetDebugInterface(IID_PPV_ARGS(&dred)))) {
            dred->SetAutoBreadcrumbsEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
            dred->SetPageFaultEnablement(D3D12_DRED_ENABLEMENT_FORCED_ON);
        }
    }
    });
    if(load_streamline)streamline_=std::make_unique<DxrStreamline>();
    check_hr(CreateDXGIFactory2(0,IID_PPV_ARGS(&factory_)),"CreateDXGIFactory2");
    for(UINT i=0;;++i) {
        ComPtr<IDXGIAdapter1> adapter;
        if(factory_->EnumAdapterByGpuPreference(i,DXGI_GPU_PREFERENCE_HIGH_PERFORMANCE,IID_PPV_ARGS(&adapter))==DXGI_ERROR_NOT_FOUND)break;
        DXGI_ADAPTER_DESC1 desc{};adapter->GetDesc1(&desc);if(desc.Flags&DXGI_ADAPTER_FLAG_SOFTWARE)continue;
        ComPtr<ID3D12Device5> candidate;
        if(FAILED(D3D12CreateDevice(adapter.Get(),D3D_FEATURE_LEVEL_12_0,IID_PPV_ARGS(&candidate))))continue;
        const auto caps=inspect_device(adapter.Get(),candidate.Get());
        if(caps.available){capabilities_=caps;device_=candidate;adapter.As(&memory_adapter_);break;}
    }
    if(!device_)throw std::runtime_error("No adapter supports DXR 1.1 and Shader Model 6.6");
    capabilities_.debug_layer_active=debug_active;capabilities_.gpu_validation_active=gpu_validation_active;
    LARGE_INTEGER driver{};
    ComPtr<IDXGIAdapter1> selected;
    if(SUCCEEDED(factory_->EnumAdapterByLuid(*reinterpret_cast<const LUID*>(&capabilities_.adapter_luid),IID_PPV_ARGS(&selected))))
        selected->CheckInterfaceSupport(__uuidof(IDXGIDevice),&driver);
    capabilities_.driver_version=std::to_string(HIWORD(driver.HighPart))+"."+std::to_string(LOWORD(driver.HighPart))+"."+std::to_string(HIWORD(driver.LowPart))+"."+std::to_string(LOWORD(driver.LowPart));
    pipeline_cache_path_=dxr_binary_directory()/"cache"/("dxr-619-"+std::to_string(capabilities_.adapter_luid)+"-"+std::to_string(driver.QuadPart)+".bin");
    D3D12_FEATURE_DATA_SHADER_CACHE cache{};
    if(SUCCEEDED(device_->CheckFeatureSupport(D3D12_FEATURE_SHADER_CACHE,&cache,sizeof(cache))) && (cache.SupportFlags&D3D12_SHADER_CACHE_SUPPORT_LIBRARY)) {
        std::ifstream saved(pipeline_cache_path_,std::ios::binary|std::ios::ate);
        if(saved && saved.tellg()>0 && saved.tellg()<256*1024*1024){pipeline_library_data_.resize(std::size_t(saved.tellg()));saved.seekg(0);if(!saved.read(reinterpret_cast<char*>(pipeline_library_data_.data()),pipeline_library_data_.size()))pipeline_library_data_.clear();}
        if(FAILED(device_->CreatePipelineLibrary(pipeline_library_data_.data(),pipeline_library_data_.size(),IID_PPV_ARGS(&pipeline_library_)))) {
            pipeline_library_data_.clear();device_->CreatePipelineLibrary(nullptr,0,IID_PPV_ARGS(&pipeline_library_));
        }
    }
    if(streamline_){streamline_->bind_device(device_.Get(),capabilities_);streamline_->upgrade_factory(factory_);}
    D3D12_COMMAND_QUEUE_DESC q{};q.Type=D3D12_COMMAND_LIST_TYPE_DIRECT;
    if(streamline_)streamline_->create_queue(device_.Get(),q,queue_);
    else check_hr(device_->CreateCommandQueue(&q,IID_PPV_ARGS(&queue_)),"CreateCommandQueue");
    for(auto& f:frames_)check_hr(device_->CreateCommandAllocator(q.Type,IID_PPV_ARGS(&f.allocator)),"CreateCommandAllocator");
    check_hr(device_->CreateCommandList(0,q.Type,frames_[0].allocator.Get(),nullptr,IID_PPV_ARGS(&list_)),"CreateCommandList");
    if(capabilities_.enhanced_barriers && !std::getenv("DXR_LEGACY_BARRIERS"))check_hr(list_.As(&enhanced_list_),"Query enhanced barrier command list");
    check_hr(list_->Close(),"Close initial list");
    check_hr(device_->CreateFence(0,D3D12_FENCE_FLAG_NONE,IID_PPV_ARGS(&fence_)),"CreateFence");
    event_=CreateEventW(nullptr,FALSE,FALSE,nullptr);
    if(!event_)throw std::runtime_error("CreateEvent failed");
    descriptors_=std::make_shared<D3d12DescriptorPool>();
    D3D12_DESCRIPTOR_HEAP_DESC h{};h.Type=D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;h.NumDescriptors=descriptors_->capacity;h.Flags=D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    check_hr(device_->CreateDescriptorHeap(&h,IID_PPV_ARGS(&descriptors_->heap)),"CreateDescriptorHeap");
    descriptors_->stride=device_->GetDescriptorHandleIncrementSize(h.Type);
    h.Type=D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER;h.NumDescriptors=18;
    check_hr(device_->CreateDescriptorHeap(&h,IID_PPV_ARGS(&samplers_)),"Create sampler heap");
    for(UINT nearest=0;nearest<2;++nearest)for(UINT s=0;s<3;++s)for(UINT t=0;t<3;++t) {
        D3D12_SAMPLER_DESC sampler{};sampler.Filter=nearest?D3D12_FILTER_MIN_MAG_MIP_POINT:D3D12_FILTER_MIN_MAG_MIP_LINEAR;
        const D3D12_TEXTURE_ADDRESS_MODE modes[]={D3D12_TEXTURE_ADDRESS_MODE_WRAP,D3D12_TEXTURE_ADDRESS_MODE_MIRROR,D3D12_TEXTURE_ADDRESS_MODE_CLAMP};
        sampler.AddressU=modes[s];sampler.AddressV=modes[t];sampler.AddressW=D3D12_TEXTURE_ADDRESS_MODE_WRAP;
        sampler.MaxLOD=D3D12_FLOAT32_MAX;sampler.ComparisonFunc=D3D12_COMPARISON_FUNC_ALWAYS;
        auto handle=samplers_->GetCPUDescriptorHandleForHeapStart();handle.ptr+=(nearest*9+s*3+t)*device_->GetDescriptorHandleIncrementSize(h.Type);
        device_->CreateSampler(&sampler,handle);
    }
    D3D12_QUERY_HEAP_DESC queries{};queries.Type=D3D12_QUERY_HEAP_TYPE_TIMESTAMP;queries.Count=3*timestamp_count;
    check_hr(device_->CreateQueryHeap(&queries,IID_PPV_ARGS(&queries_)),"Create timestamp heap");
    query_readback_=readback_buffer(3*timestamp_count*sizeof(std::uint64_t));
    check_hr(queue_->GetTimestampFrequency(&frequency_),"GetTimestampFrequency");
}
D3d12Context::~D3d12Context(){try{flush();}catch(...){}save_pipeline_cache();if(streamline_)streamline_->shutdown();queue_.Reset();factory_.Reset();streamline_.reset();if(event_)CloseHandle(event_);}
ID3D12GraphicsCommandList4* D3d12Context::begin() {
    if(recording_)submit();
    frame_=(frame_+1)%3;auto& f=frames_[frame_];wait(f.fence);
    if(f.fence) {
        std::uint64_t* data=nullptr;const D3D12_RANGE range{frame_*timestamp_count*sizeof(std::uint64_t),(frame_+1)*timestamp_count*sizeof(std::uint64_t)};
        check_hr(query_readback_->resource->Map(0,&range,reinterpret_cast<void**>(&data)),"Map timestamps");
        for(UINT i=1;i<timestamp_count;++i)if(f.stamps[i]&&f.stamps[i-1])timings_[i]=float(double(data[frame_*timestamp_count+i]-data[frame_*timestamp_count+i-1])*1000.0/double(frequency_));
        if(f.stamps[0]&&f.stamps[timestamp_count-1]) {
            timings_[0]=float(double(data[frame_*timestamp_count+timestamp_count-1]-data[frame_*timestamp_count])*1000.0/double(frequency_));
            completed_timing_tag_=f.timing_tag;
        }
        const D3D12_RANGE empty{0,0};query_readback_->resource->Unmap(0,&empty);
    }
    f.retained.clear();f.stamps.fill(false);f.timing_tag=0;
    for(auto& page:f.uploads)page.used=0;
    check_hr(f.allocator->Reset(),"Reset allocator");check_hr(list_->Reset(f.allocator.Get(),nullptr),"Reset command list");
    ID3D12DescriptorHeap* heaps[]={descriptors_->heap.Get(),samplers_.Get()};list_->SetDescriptorHeaps(2,heaps);
    recording_=true;return list_.Get();
}
std::uint64_t D3d12Context::submit() {
    if(!recording_)return fence_value_;
    auto& f=frames_[frame_];
    for(UINT i=0;i<timestamp_count;++i)if(f.stamps[i])list_->ResolveQueryData(queries_.Get(),D3D12_QUERY_TYPE_TIMESTAMP,frame_*timestamp_count+i,1,query_readback_->resource.Get(),(frame_*timestamp_count+i)*sizeof(std::uint64_t));
    check_hr(list_->Close(),"Close command list");ID3D12CommandList* lists[]={list_.Get()};queue_->ExecuteCommandLists(1,lists);
    recording_=false;check_hr(queue_->Signal(fence_.Get(),++fence_value_),"Signal fence");f.fence=fence_value_;return fence_value_;
}
std::uint64_t D3d12Context::signal_presented() {
    check_hr(queue_->Signal(fence_.Get(),++fence_value_),"Signal presented frame");
    frames_[frame_].fence=fence_value_;return fence_value_;
}
void D3d12Context::wait(std::uint64_t value) {
    const auto completed=fence_->GetCompletedValue();
    if(completed==UINT64_MAX)check_hr(device_->GetDeviceRemovedReason(),"DXR device removed");
    if(completed>=value)return;
    check_hr(fence_->SetEventOnCompletion(value,event_),"SetEventOnCompletion");
    if(WaitForSingleObject(event_,30000)!=WAIT_OBJECT_0)throw std::runtime_error("DXR GPU fence timed out");
    check_hr(device_->GetDeviceRemovedReason(),"DXR execution");
}
void D3d12Context::flush(){if(fence_){wait(submit());for(auto& frame:frames_)frame.retained.clear();}}
void D3d12Context::retain(const D3d12ResourcePtr& resource){if(resource&&recording_)frames_[frame_].retained.push_back(resource);}
void D3d12Context::transition(const D3d12ResourcePtr& resource,D3D12_RESOURCE_STATES state) {
    retain(resource);if(resource->state==state)return;
    if(enhanced_list_) {
        const auto before=barrier_scope(resource->state),after=barrier_scope(state);
        D3D12_BARRIER_GROUP group{};group.NumBarriers=1;
        if(resource->resource->GetDesc().Dimension==D3D12_RESOURCE_DIMENSION_BUFFER) {
            D3D12_BUFFER_BARRIER barrier{};barrier.SyncBefore=barrier.SyncAfter=D3D12_BARRIER_SYNC_ALL;
            barrier.AccessBefore=before.access;barrier.AccessAfter=after.access;barrier.pResource=resource->resource.Get();barrier.Size=UINT64_MAX;
            group.Type=D3D12_BARRIER_TYPE_BUFFER;group.pBufferBarriers=&barrier;enhanced_list_->Barrier(1,&group);
        }else {
            D3D12_TEXTURE_BARRIER barrier{};barrier.SyncBefore=barrier.SyncAfter=D3D12_BARRIER_SYNC_ALL;
            barrier.AccessBefore=before.access;barrier.AccessAfter=after.access;barrier.LayoutBefore=before.layout;barrier.LayoutAfter=after.layout;
            barrier.pResource=resource->resource.Get();barrier.Subresources.IndexOrFirstMipLevel=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
            group.Type=D3D12_BARRIER_TYPE_TEXTURE;group.pTextureBarriers=&barrier;enhanced_list_->Barrier(1,&group);
        }
        resource->state=state;return;
    }
    D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;b.Transition.pResource=resource->resource.Get();
    b.Transition.StateBefore=resource->state;b.Transition.StateAfter=state;b.Transition.Subresource=D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
    list_->ResourceBarrier(1,&b);resource->state=state;
}
void D3d12Context::uav_barrier(const D3d12ResourcePtr& resource){
    retain(resource);
    if(enhanced_list_) {
        D3D12_GLOBAL_BARRIER barrier{};barrier.SyncBefore=barrier.SyncAfter=D3D12_BARRIER_SYNC_ALL;
        barrier.AccessBefore=D3D12_BARRIER_ACCESS_UNORDERED_ACCESS|D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_WRITE;
        barrier.AccessAfter=barrier.AccessBefore|D3D12_BARRIER_ACCESS_RAYTRACING_ACCELERATION_STRUCTURE_READ|D3D12_BARRIER_ACCESS_SHADER_RESOURCE;
        D3D12_BARRIER_GROUP group{};group.Type=D3D12_BARRIER_TYPE_GLOBAL;group.NumBarriers=1;group.pGlobalBarriers=&barrier;enhanced_list_->Barrier(1,&group);return;
    }
    D3D12_RESOURCE_BARRIER b{};b.Type=D3D12_RESOURCE_BARRIER_TYPE_UAV;b.UAV.pResource=resource?resource->resource.Get():nullptr;list_->ResourceBarrier(1,&b);
}
std::shared_ptr<D3d12Descriptor> D3d12Context::descriptor(UINT count){auto d=std::make_shared<D3d12Descriptor>();d->pool=descriptors_;d->count=count;d->index=descriptors_->allocate(count);return d;}
D3d12ResourcePtr D3d12Context::pooled_resource(const D3D12_RESOURCE_DESC& desc,UINT stride,D3D12_RESOURCE_STATES initial) {
    const D3d12ResourcePool::Key key{UINT(desc.Dimension),desc.Width,desc.Height,UINT(desc.Format),desc.MipLevels,UINT(desc.Flags),stride,initial==D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE};
    D3d12Resource* resource=nullptr;
    {
        std::lock_guard lock(resource_pool_->mutex);
        for(auto it=resource_pool_->entries.begin();it!=resource_pool_->entries.end();++it)if(it->key==key) {
            resource=it->resource.release();resource_pool_->bytes-=resource->allocation_bytes;resource_pool_->entries.erase(it);++resource_reuses_;break;
        }
    }
    if(!resource){resource=new D3d12Resource;resource->state=initial;}
    std::weak_ptr<D3d12ResourcePool> pool=resource_pool_;
    D3d12ResourcePtr result(resource,[pool,key](D3d12Resource* value){if(auto p=pool.lock())p->recycle(key,value);else delete value;});
    if(!resource->resource) {
        const auto heap=heap_properties(D3D12_HEAP_TYPE_DEFAULT);
        check_hr(device_->CreateCommittedResource(&heap,D3D12_HEAP_FLAG_NONE,&desc,initial,nullptr,IID_PPV_ARGS(&resource->resource)),"Create pooled GPU resource");
        resource->allocation_bytes=device_->GetResourceAllocationInfo(0,1,&desc).SizeInBytes;
        resource->bytes=desc.Dimension==D3D12_RESOURCE_DIMENSION_BUFFER?desc.Width:resource->allocation_bytes;
        resource->accounting=allocated_bytes_;allocated_bytes_->fetch_add(resource->allocation_bytes);++resource_creations_;
    }else if(desc.Dimension==D3D12_RESOURCE_DIMENSION_BUFFER && resource->state!=initial) {
        if(!recording_)begin();transition(result,initial);
    }
    retain(result);return result;
}
std::uint64_t D3d12Context::pooled_bytes() const {std::lock_guard lock(resource_pool_->mutex);return resource_pool_->bytes;}
D3d12ResourcePtr D3d12Context::buffer(std::uint64_t bytes,UINT stride,D3D12_RESOURCE_FLAGS flags,D3D12_RESOURCE_STATES initial) {
    const auto desc=buffer_desc(bytes,flags);auto r=pooled_resource(desc,stride,initial);
    if(stride && !r->srv) {
        r->srv=descriptor();D3D12_SHADER_RESOURCE_VIEW_DESC s{};s.ViewDimension=D3D12_SRV_DIMENSION_BUFFER;s.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
        s.Buffer.NumElements=static_cast<UINT>(desc.Width/stride);s.Buffer.StructureByteStride=stride;device_->CreateShaderResourceView(r->resource.Get(),&s,descriptors_->cpu(r->srv_index()));
        if(flags&D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS) {
            r->uav=descriptor();D3D12_UNORDERED_ACCESS_VIEW_DESC u{};u.ViewDimension=D3D12_UAV_DIMENSION_BUFFER;u.Buffer.NumElements=s.Buffer.NumElements;u.Buffer.StructureByteStride=stride;
            device_->CreateUnorderedAccessView(r->resource.Get(),nullptr,&u,descriptors_->cpu(r->uav_index()));
        }
    }
    retain(r);return r;
}
D3d12ResourcePtr D3d12Context::upload(std::span<const std::byte> bytes) {
    auto r=std::make_shared<D3d12Resource>();r->bytes=std::max<std::uint64_t>(bytes.size(),256);r->state=D3D12_RESOURCE_STATE_GENERIC_READ;
    const auto h=heap_properties(D3D12_HEAP_TYPE_UPLOAD);const auto d=buffer_desc(r->bytes,D3D12_RESOURCE_FLAG_NONE);
    check_hr(device_->CreateCommittedResource(&h,D3D12_HEAP_FLAG_NONE,&d,r->state,nullptr,IID_PPV_ARGS(&r->resource)),"Create upload buffer");
    r->allocation_bytes=device_->GetResourceAllocationInfo(0,1,&d).SizeInBytes;r->accounting=allocated_bytes_;allocated_bytes_->fetch_add(r->allocation_bytes);++resource_creations_;
    void* data=nullptr;const D3D12_RANGE empty{0,0};check_hr(r->resource->Map(0,&empty,&data),"Map upload");
    if(!bytes.empty())std::memcpy(data,bytes.data(),bytes.size());r->resource->Unmap(0,nullptr);retain(r);return r;
}
D3d12UploadSlice D3d12Context::upload_frame(std::span<const std::byte> bytes,UINT64 alignment) {
    if(!recording_)throw std::logic_error("frame upload requires a recording command list");
    if(!alignment || (alignment&(alignment-1)))throw std::invalid_argument("upload alignment must be a power of two");
    constexpr UINT64 page_size=1024*1024;
    if(bytes.size()>page_size/2)return {upload(bytes),0};
    auto& pages=frames_[frame_].uploads;
    for(auto& page:pages) {
        const auto offset=(page.used+alignment-1)&~(alignment-1);
        if(offset+bytes.size()<=page_size){if(!bytes.empty())std::memcpy(page.data+offset,bytes.data(),bytes.size());page.used=offset+std::max<std::size_t>(bytes.size(),1);return {page.buffer,offset};}
    }
    if(pages.size()>=8)return {upload(bytes),0};
    UploadPage page;std::vector<std::byte> empty(page_size);page.buffer=upload(empty);
    const D3D12_RANGE no_read{0,0};check_hr(page.buffer->resource->Map(0,&no_read,reinterpret_cast<void**>(&page.data)),"Map frame upload arena");
    if(!bytes.empty())std::memcpy(page.data,bytes.data(),bytes.size());page.used=std::max<std::size_t>(bytes.size(),1);
    pages.push_back(page);return {page.buffer,0};
}
D3d12ResourcePtr D3d12Context::upload_buffer(std::span<const std::byte> bytes,UINT stride) {
    auto r=buffer(std::max<std::size_t>(bytes.size(),stride),stride,D3D12_RESOURCE_FLAG_NONE,D3D12_RESOURCE_STATE_COPY_DEST);
    if(!bytes.empty()){auto staging=upload_frame(bytes);list_->CopyBufferRegion(r->resource.Get(),0,staging.buffer->resource.Get(),staging.offset,bytes.size());}
    transition(r,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);return r;
}
D3d12ResourcePtr D3d12Context::texture(UINT w,UINT h,DXGI_FORMAT format,UINT mips,bool storage) {
    D3D12_RESOURCE_DESC d{};
    d.Dimension=D3D12_RESOURCE_DIMENSION_TEXTURE2D;d.Width=w;d.Height=h;d.DepthOrArraySize=1;d.MipLevels=static_cast<UINT16>(mips);d.Format=format;d.SampleDesc.Count=1;
    d.Flags=storage?D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS:D3D12_RESOURCE_FLAG_NONE;
    auto r=pooled_resource(d,0,D3D12_RESOURCE_STATE_COMMON);if(r->srv)return r;
    r->srv=descriptor();D3D12_SHADER_RESOURCE_VIEW_DESC s{};s.Format=format;s.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;s.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;s.Texture2D.MipLevels=mips;
    device_->CreateShaderResourceView(r->resource.Get(),&s,descriptors_->cpu(r->srv_index()));
    if(storage){r->uav=descriptor();D3D12_UNORDERED_ACCESS_VIEW_DESC u{};u.Format=format;u.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE2D;device_->CreateUnorderedAccessView(r->resource.Get(),nullptr,&u,descriptors_->cpu(r->uav_index()));}
    retain(r);return r;
}
D3d12ResourcePtr D3d12Context::upload_texture(UINT w,UINT h,DXGI_FORMAT format,UINT bpp,std::span<const std::byte> bytes) {
    if(bytes.size()!=std::size_t(w)*h*bpp)throw std::runtime_error("invalid DXR texture data size");
    auto r=texture(w,h,format,1,false);transition(r,D3D12_RESOURCE_STATE_COPY_DEST);
    const auto desc=r->resource->GetDesc();D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{};UINT64 total=0;
    device_->GetCopyableFootprints(&desc,0,1,0,&layout,nullptr,nullptr,&total);
    std::vector<std::byte> packed(static_cast<std::size_t>(total));
    for(UINT y=0;y<h;++y)std::memcpy(packed.data()+layout.Offset+std::size_t(y)*layout.Footprint.RowPitch,bytes.data()+std::size_t(y)*w*bpp,std::size_t(w)*bpp);
    auto staging=upload_frame(packed,D3D12_TEXTURE_DATA_PLACEMENT_ALIGNMENT);D3D12_TEXTURE_COPY_LOCATION from{};from.pResource=staging.buffer->resource.Get();from.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;from.PlacedFootprint=layout;from.PlacedFootprint.Offset+=staging.offset;
    D3D12_TEXTURE_COPY_LOCATION to{};to.pResource=r->resource.Get();to.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    list_->CopyTextureRegion(&to,0,0,0,&from,nullptr);transition(r,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);return r;
}
D3d12ResourcePtr D3d12Context::readback_buffer(std::uint64_t bytes) {
    auto r=std::make_shared<D3d12Resource>();r->bytes=bytes;r->state=D3D12_RESOURCE_STATE_COPY_DEST;
    const auto h=heap_properties(D3D12_HEAP_TYPE_READBACK);const auto d=buffer_desc(bytes,D3D12_RESOURCE_FLAG_NONE);
    check_hr(device_->CreateCommittedResource(&h,D3D12_HEAP_FLAG_NONE,&d,r->state,nullptr,IID_PPV_ARGS(&r->resource)),"Create readback buffer");
    r->allocation_bytes=device_->GetResourceAllocationInfo(0,1,&d).SizeInBytes;r->accounting=allocated_bytes_;allocated_bytes_->fetch_add(r->allocation_bytes);++resource_creations_;retain(r);return r;
}
ComPtr<ID3D12PipelineState> D3d12Context::compute_pipeline(const wchar_t* name,const D3D12_COMPUTE_PIPELINE_STATE_DESC& desc) {
    // The driver validates the complete PSO (including its root signature) when
    // loading. Shader identity keeps rebuilt bytecode from reusing an old entry.
    UINT64 hash=14695981039346656037ull;const auto* bytes=static_cast<const unsigned char*>(desc.CS.pShaderBytecode);
    for(SIZE_T i=0;i<desc.CS.BytecodeLength;++i){hash^=bytes[i];hash*=1099511628211ull;}
    const std::wstring key=std::wstring(name)+L"-"+std::to_wstring(hash);
    ComPtr<ID3D12PipelineState> pipeline;
    if(pipeline_library_ && SUCCEEDED(pipeline_library_->LoadComputePipeline(key.c_str(),&desc,IID_PPV_ARGS(&pipeline)))){++pipeline_cache_hits_;return pipeline;}
    check_hr(device_->CreateComputePipelineState(&desc,IID_PPV_ARGS(&pipeline)),"Create cached compute pipeline");
    if(pipeline_library_ && SUCCEEDED(pipeline_library_->StorePipeline(key.c_str(),pipeline.Get())))pipeline_cache_dirty_=true;
    return pipeline;
}
void D3d12Context::save_pipeline_cache() noexcept {
    if(!pipeline_library_ || !pipeline_cache_dirty_)return;
    try {
        std::vector<std::byte> data(pipeline_library_->GetSerializedSize());
        if(FAILED(pipeline_library_->Serialize(data.data(),data.size())))return;
        std::filesystem::create_directories(pipeline_cache_path_.parent_path());auto temporary=pipeline_cache_path_;temporary+=L"."+std::to_wstring(GetCurrentProcessId())+L".tmp";
        {std::ofstream saved(temporary,std::ios::binary|std::ios::trunc);saved.write(reinterpret_cast<const char*>(data.data()),data.size());if(!saved)return;}
        MoveFileExW(temporary.c_str(),pipeline_cache_path_.c_str(),MOVEFILE_REPLACE_EXISTING|MOVEFILE_WRITE_THROUGH);
    }catch(...){} // A read-only cache directory must never prevent rendering.
}
void D3d12Context::timestamp(UINT stage){if(stage>=timestamp_count)throw std::out_of_range("DXR timestamp");frames_[frame_].stamps[stage]=true;list_->EndQuery(queries_.Get(),D3D12_QUERY_TYPE_TIMESTAMP,frame_*timestamp_count+stage);}
UINT D3d12Context::sampler(bool nearest,unsigned s,unsigned t) const {return (nearest?9:0)+std::min(s,2u)*3+std::min(t,2u);}
bool D3d12Context::query_video_memory(std::uint64_t& usage,std::uint64_t& budget) const {
    usage=budget=0;DXGI_QUERY_VIDEO_MEMORY_INFO info{};
    if(!memory_adapter_ || FAILED(memory_adapter_->QueryVideoMemoryInfo(0,DXGI_MEMORY_SEGMENT_GROUP_LOCAL,&info)))return false;
    usage=info.CurrentUsage;budget=info.Budget;return true;
}
DxrDeviceCapabilities query_dxr_capabilities() {
    static const auto result=[] {try{return D3d12Context::create(false)->capabilities();}catch(const std::exception& e){DxrDeviceCapabilities c;c.reason=e.what();return c;}}();return result;
}
bool dxr_available(std::string* reason){const auto c=query_dxr_capabilities();if(reason)*reason=c.reason;return c.available;}
void D3d12Context::check_validation() const {
    ComPtr<ID3D12InfoQueue> queue;
    if(FAILED(device_.As(&queue)))return;
    std::string errors;
    for(UINT64 i=0;i<queue->GetNumStoredMessagesAllowedByRetrievalFilter();++i) {
        SIZE_T size=0;queue->GetMessage(i,nullptr,&size);std::vector<std::byte> storage(size);
        auto* message=reinterpret_cast<D3D12_MESSAGE*>(storage.data());
        if(SUCCEEDED(queue->GetMessage(i,message,&size)) && message->Severity<=D3D12_MESSAGE_SEVERITY_ERROR) {
            errors.append(message->pDescription,message->DescriptionByteLength);errors+='\n';
        }
    }
    queue->ClearStoredMessages();
    if(!errors.empty())throw std::runtime_error("D3D12 validation errors:\n"+errors);
}
} // namespace renderer
