#include "render/dxr/dxr_streamline.h"
#include "core/math/types.h"
#include <sl.h>
#include <sl_dlss.h>
#include <sl_dlss_d.h>
#include <sl_security.h>
#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <bcrypt.h>
#include <psapi.h>
#include <fstream>
#include <dxr_runtime_manifest.h>

namespace renderer {
namespace {
constexpr const char* project_id="cef787f2-5072-415d-898e-12dcf3001821";
// Streamline owns process-global plugin state; hardware probes must not rebind it.
std::mutex streamline_mutex;
bool streamline_in_use=false;
std::filesystem::path module_path(HMODULE module) {
    std::wstring path(32768,L'\0');auto size=GetModuleFileNameW(module,path.data(),DWORD(path.size()));
    if(!size || size>=path.size())throw std::runtime_error("Cannot inspect loaded DLSS module");path.resize(size);return path;
}
std::string file_sha256(const std::filesystem::path& path) {
    struct Hash {
        BCRYPT_ALG_HANDLE algorithm=nullptr;BCRYPT_HASH_HANDLE hash=nullptr;
        ~Hash(){if(hash)BCryptDestroyHash(hash);if(algorithm)BCryptCloseAlgorithmProvider(algorithm,0);}
    } state;
    auto check=[](NTSTATUS code){if(code<0)throw std::runtime_error("SHA-256 runtime verification failed");};
    check(BCryptOpenAlgorithmProvider(&state.algorithm,BCRYPT_SHA256_ALGORITHM,nullptr,0));
    check(BCryptCreateHash(state.algorithm,&state.hash,nullptr,0,nullptr,0,0));
    std::ifstream input(path,std::ios::binary);if(!input)throw std::runtime_error("Cannot read loaded DLSS module");
    std::array<unsigned char,65536> block;
    while(input){input.read(reinterpret_cast<char*>(block.data()),block.size());if(input.gcount())check(BCryptHashData(state.hash,block.data(),ULONG(input.gcount()),0));}
    if(!input.eof())throw std::runtime_error("Cannot hash loaded DLSS module");
    std::array<unsigned char,32> digest;check(BCryptFinishHash(state.hash,digest.data(),ULONG(digest.size()),0));
    constexpr const char* hex="0123456789abcdef";std::string result;
    for(auto value:digest){result+=hex[value>>4];result+=hex[value&15];}return result;
}
std::string file_version(const std::filesystem::path& path) {
    DWORD ignored=0;const auto size=GetFileVersionInfoSizeW(path.c_str(),&ignored);if(!size)return {};
    std::vector<std::byte> data(size);if(!GetFileVersionInfoW(path.c_str(),0,size,data.data()))return {};
    VS_FIXEDFILEINFO* info=nullptr;UINT length=0;
    if(!VerQueryValueW(data.data(),L"\\",reinterpret_cast<void**>(&info),&length) || length<sizeof(*info))return {};
    return std::to_string(HIWORD(info->dwFileVersionMS))+"."+std::to_string(LOWORD(info->dwFileVersionMS))+"."+std::to_string(HIWORD(info->dwFileVersionLS))+"."+std::to_string(LOWORD(info->dwFileVersionLS));
}
sl::float4x4 sl_matrix(const Mat4& m) {
    // The SDK multiplies row vectors, while renderer matrices multiply columns.
    sl::float4x4 result;for(unsigned row=0;row<4;++row)for(unsigned column=0;column<4;++column)reinterpret_cast<float*>(&result[row])[column]=m(column,row);return result;
}
Mat4 view_matrix(const DxrFloat4& eye,const DxrFloat4& f,const DxrFloat4& r,const DxrFloat4& u) {
    Mat4 v=Mat4::Identity();v.row(0)=Vec4(r.x,r.y,r.z,-r.x*eye.x-r.y*eye.y-r.z*eye.z);
    v.row(1)=Vec4(u.x,u.y,u.z,-u.x*eye.x-u.y*eye.y-u.z*eye.z);v.row(2)=Vec4(-f.x,-f.y,-f.z,f.x*eye.x+f.y*eye.y+f.z*eye.z);return v;
}
Mat4 projection_matrix(const DxrFloat4& r,const DxrFloat4& u) {
    Mat4 p=Mat4::Zero();p(0,0)=2/r.w;p(1,1)=2/u.w;p(2,2)=-1;p(2,3)=-.05f;p(3,2)=-1;return p;
}
}
struct DxrStreamline::Impl {
    std::vector<DxrRuntimeModule> modules;
    bool pinned=true,inspected_ngx=false;
    void inspect_module(const char* name,HMODULE module) {
        if(!module)return;
        const auto path=module_path(module);
        for(const auto& old:modules)if(old.name==name && old.path==path.generic_string())return;
        DxrRuntimeModule record;record.name=name;record.path=path.generic_string();record.version=file_version(path);record.sha256=file_sha256(path);
        for(const auto& expected:dxr_pinned_runtimes)if(record.name==expected.name)record.expected_sha256=expected.sha256;
        record.pinned=!record.expected_sha256.empty() && record.sha256==record.expected_sha256;pinned&=record.pinned;
        modules.push_back(std::move(record));
    }
    void inspect_function(const char* name,const void* function) {
        if(!function)return;HMODULE module=nullptr;
        if(!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS|GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,reinterpret_cast<LPCWSTR>(function),&module))throw std::runtime_error("Cannot locate active DLSS plugin");
        inspect_module(name,module);
    }
    void inspect_ngx(bool after_evaluation=true) {
        if(after_evaluation && inspected_ngx)return;
        if(after_evaluation)inspected_ngx=true;
        HMODULE loaded[2048];DWORD bytes=0;
        if(!K32EnumProcessModules(GetCurrentProcess(),loaded,sizeof(loaded),&bytes))throw std::runtime_error("Cannot enumerate DLSS runtimes");
        for(unsigned j=0;j<std::min<unsigned>(bytes/sizeof(HMODULE),std::size(loaded));++j) {
            auto path=module_path(loaded[j]);auto name=path.filename().string();
            for(char& c:name)c=char(std::tolower(static_cast<unsigned char>(c)));
            auto location=path.generic_string();
            for(char& c:location)c=char(std::tolower(static_cast<unsigned char>(c)));
            // Driver overrides use hashed DLL names, including sl.common.
            // Inspect the loaded image, not the unused file in our SDK folder.
            if(location.find("/sl_common_override_")!=std::string::npos)name="sl.common.dll";
            if(name=="nvngx_dlss.dll" || name=="nvngx_dlssd.dll" || name=="sl.common.dll")inspect_module(name.c_str(),loaded[j]);
        }
    }
    HMODULE module=nullptr;
    bool initialized=false,owns_plugins=false,presentation=false,sr=false,rr=false;
    bool sr_failed=false,rr_failed=false;
    sl::ViewportHandle viewport{0};
    std::string reason;
    std::uint64_t memory=0;
    // Command-queue proxies retain a raw device-proxy pointer in Streamline.
    ComPtr<ID3D12Device> device_proxy;
    PFun_slInit* init=nullptr;PFun_slShutdown* shutdown=nullptr;PFun_slSetD3DDevice* set_device=nullptr;
    PFun_slIsFeatureSupported* supported=nullptr;PFun_slGetFeatureFunction* feature_function=nullptr;
    PFun_slUpgradeInterface* upgrade=nullptr;PFun_slGetNewFrameToken* frame_token=nullptr;
    PFun_slSetConstants* constants=nullptr;PFun_slSetTagForFrame* tags=nullptr;PFun_slEvaluateFeature* evaluate=nullptr;
    PFun_slFreeResources* free_resources=nullptr;
    PFun_slDLSSSetOptions* sr_options=nullptr;PFun_slDLSSDSetOptions* rr_options=nullptr;
    PFun_slDLSSGetOptimalSettings* sr_size=nullptr;PFun_slDLSSDGetOptimalSettings* rr_size=nullptr;
    PFun_slDLSSGetState* sr_state=nullptr;PFun_slDLSSDGetState* rr_state=nullptr;
    bool check(sl::Result result,const char* operation) {
        if(result!=sl::Result::eOk && std::getenv("DXR_VALIDATE"))std::fprintf(stderr,"[DXR Streamline] %s: %u\n",operation,unsigned(result));
        if(result==sl::Result::eOk)return true;reason=std::string(operation)+" (Streamline result "+std::to_string(unsigned(result))+")";return false;
    }
    template<class T> void load(T*& function,const char* name) {
        function=reinterpret_cast<T*>(GetProcAddress(module,name));if(!function)throw std::runtime_error(std::string("Streamline export missing: ")+name);
    }
    template<class T> bool feature(T*& function,sl::Feature id,const char* name) {
        void* address=nullptr;if(!check(feature_function(id,name,address),name))return false;function=reinterpret_cast<T*>(address);return function!=nullptr;
    }
    ~Impl() {
        if(initialized)shutdown();
        device_proxy.Reset();
        if(module)FreeLibrary(module);
        if(owns_plugins){std::lock_guard lock(streamline_mutex);streamline_in_use=false;}
    }
};
DxrStreamline::DxrStreamline():impl_(std::make_unique<Impl>()) {
    auto& i=*impl_;
    if(std::getenv("DXR_DISABLE_DLSS")){i.reason="DLSS disabled by DXR_DISABLE_DLSS";return;}
    {std::lock_guard lock(streamline_mutex);if(streamline_in_use){i.reason="Streamline is already bound to another DXR device";return;}streamline_in_use=true;i.owns_plugins=true;}
    const auto directory=dxr_binary_directory()/"Streamline",path=directory/"sl.interposer.dll";
    if(!std::filesystem::exists(path)){i.reason="Streamline 2.14.1 runtime is missing";return;}
    if(!sl::security::verifyEmbeddedSignature(path.c_str())){i.reason="Streamline runtime signature verification failed";return;}
    i.module=LoadLibraryExW(path.c_str(),nullptr,LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR|LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if(!i.module){i.reason="Cannot load Streamline 2.14.1: Win32 error "+std::to_string(GetLastError());return;}
    try {
        i.load(i.init,"slInit");i.load(i.shutdown,"slShutdown");i.load(i.set_device,"slSetD3DDevice");i.load(i.supported,"slIsFeatureSupported");
        i.load(i.feature_function,"slGetFeatureFunction");i.load(i.upgrade,"slUpgradeInterface");i.load(i.frame_token,"slGetNewFrameToken");
        i.load(i.constants,"slSetConstants");i.load(i.tags,"slSetTagForFrame");i.load(i.evaluate,"slEvaluateFeature");i.load(i.free_resources,"slFreeResources");
        const auto plugin_path=directory.wstring();const wchar_t* paths[]={plugin_path.c_str()};const sl::Feature features[]={sl::kFeatureDLSS,sl::kFeatureDLSS_RR};
        sl::Preferences preferences;preferences.pathsToPlugins=paths;preferences.numPathsToPlugins=1;preferences.featuresToLoad=features;preferences.numFeaturesToLoad=2;
        preferences.engine=sl::EngineType::eCustom;preferences.engineVersion="3drenderer DXR 1.0";preferences.projectId=project_id;
        preferences.renderAPI=sl::RenderAPI::eD3D12;
        // Fixed local runtime only: neither OTA downloads nor downloaded plugin loading is enabled.
        preferences.flags=sl::PreferenceFlags::eDisableCLStateTracking|sl::PreferenceFlags::eUseManualHooking|sl::PreferenceFlags::eUseDXGIFactoryProxy|sl::PreferenceFlags::eUseFrameBasedResourceTagging;
        preferences.logLevel=sl::LogLevel::eDefault;preferences.showConsole=false;
        preferences.logMessageCallback=[](sl::LogType,const char* message){if(std::getenv("DXR_VALIDATE"))std::fprintf(stderr,"[Streamline] %s\n",message);};
        i.initialized=i.check(i.init(preferences,sl::kSDKVersion),"slInit");
        i.inspect_module("sl.interposer.dll",i.module);
    }catch(const std::exception& e){i.reason=e.what();}
}
DxrStreamline::~DxrStreamline()=default;
void DxrStreamline::shutdown(){auto& i=*impl_;if(i.initialized){i.shutdown();i.initialized=false;}}
void DxrStreamline::release_resources(D3d12Context& context) {
    auto& i=*impl_;if(!i.initialized || !i.memory)return;
    context.flush();
    if(i.rr)i.check(i.free_resources(sl::kFeatureDLSS_RR,i.viewport),"Release DLSS RR");
    if(i.sr)i.check(i.free_resources(sl::kFeatureDLSS,i.viewport),"Release DLSS SR");
    i.memory=0;
}
void DxrStreamline::bind_device(ID3D12Device* device,DxrDeviceCapabilities& caps) {
    auto& i=*impl_;if(!i.initialized)return;
    if(!i.check(i.set_device(device),"slSetD3DDevice"))return;
    sl::AdapterInfo adapter;adapter.deviceLUID=reinterpret_cast<std::uint8_t*>(&caps.adapter_luid);adapter.deviceLUIDSizeInBytes=sizeof(caps.adapter_luid);
    i.sr=i.check(i.supported(sl::kFeatureDLSS,adapter),"DLSS SR unavailable") && i.feature(i.sr_options,sl::kFeatureDLSS,"slDLSSSetOptions") && i.feature(i.sr_size,sl::kFeatureDLSS,"slDLSSGetOptimalSettings") && i.feature(i.sr_state,sl::kFeatureDLSS,"slDLSSGetState");
    i.rr=i.check(i.supported(sl::kFeatureDLSS_RR,adapter),"DLSS RR unavailable") && i.feature(i.rr_options,sl::kFeatureDLSS_RR,"slDLSSDSetOptions") && i.feature(i.rr_size,sl::kFeatureDLSS_RR,"slDLSSDGetOptimalSettings") && i.feature(i.rr_state,sl::kFeatureDLSS_RR,"slDLSSDGetState");
    caps.dlss_sr_supported=i.sr;caps.dlss_rr_supported=i.rr;
    if(i.rr && i.sr)i.reason.clear();
    try {
        i.inspect_function("sl.dlss.dll",reinterpret_cast<const void*>(i.sr_options));
        i.inspect_function("sl.dlss_d.dll",reinterpret_cast<const void*>(i.rr_options));
        i.inspect_ngx(false);
        if(!i.pinned){i.sr_failed=i.rr_failed=true;i.reason="Driver replaced the pinned Streamline plugins; DLSS disabled for reproducibility (see runtime module hashes)";}
    }catch(const std::exception& error){i.pinned=false;i.sr_failed=i.rr_failed=true;i.reason=error.what();}
}
void DxrStreamline::upgrade_factory(ComPtr<IDXGIFactory6>& factory) {
    auto& i=*impl_;if(!i.initialized)return;
    void* value=factory.Get();const auto result=i.upgrade(&value);
    if(value!=factory.Get()){factory.Reset();factory.Attach(static_cast<IDXGIFactory6*>(value));}
    if(!i.check(result,"Upgrade DXGI factory")){i.sr=i.rr=false;}
}
void DxrStreamline::upgrade_swapchain(ComPtr<IDXGISwapChain1>& swapchain) {
    auto& i=*impl_;if(!i.initialized)return;
    const GUID native_guid{0xadec44e2,0x61f0,0x45c3,{0xad,0x9f,0x1b,0x37,0x37,0x92,0x84,0xff}};
    ComPtr<IUnknown> native;
    if(SUCCEEDED(swapchain->QueryInterface(native_guid,reinterpret_cast<void**>(native.GetAddressOf())))){i.presentation=true;return;}
    void* value=swapchain.Get();const auto result=i.upgrade(&value);
    if(value!=swapchain.Get()){swapchain.Reset();swapchain.Attach(static_cast<IDXGISwapChain1*>(value));}
    i.presentation=i.check(result,"Upgrade DXGI swapchain");
}
void DxrStreamline::create_queue(ID3D12Device* device,const D3D12_COMMAND_QUEUE_DESC& desc,ComPtr<ID3D12CommandQueue>& queue) {
    auto& i=*impl_;
    if(i.initialized && !i.device_proxy){void* value=device;const auto result=i.upgrade(&value);
        if(value!=device)i.device_proxy.Attach(static_cast<ID3D12Device*>(value));
        if(!i.check(result,"Upgrade D3D12 device"))i.sr=i.rr=false;}
    check_hr((i.device_proxy?i.device_proxy.Get():device)->CreateCommandQueue(&desc,IID_PPV_ARGS(&queue)),"Create DXR command queue");
    // ImGui creates its own native DXGI factory for detached viewports. Native
    // DXGI must receive the native queue, never an interposer's COM proxy.
    const GUID native_guid{0xadec44e2,0x61f0,0x45c3,{0xad,0x9f,0x1b,0x37,0x37,0x92,0x84,0xff}};
    ComPtr<ID3D12CommandQueue> native;
    if(SUCCEEDED(queue->QueryInterface(native_guid,reinterpret_cast<void**>(native.GetAddressOf()))))queue=std::move(native);
}
DxrReconstruction DxrStreamline::select(DxrReconstruction requested) const {
    auto& i=*impl_;if(requested==DxrReconstruction::Reference || requested==DxrReconstruction::NrdTaau)return requested;
    if(!i.presentation){if(i.reason.empty())i.reason="DLSS requires an active DXGI presentation lifecycle";return DxrReconstruction::NrdTaau;}
    if(requested!=DxrReconstruction::NrdDlss && i.rr && !i.rr_failed)return DxrReconstruction::DlssRr;
    if(i.sr && !i.sr_failed)return DxrReconstruction::NrdDlss;
    return DxrReconstruction::NrdTaau;
}
std::pair<UINT,UINT> DxrStreamline::render_size(DxrReconstruction mode,UINT w,UINT h) {
    auto& i=*impl_;
    if(mode==DxrReconstruction::DlssRr) {
        sl::DLSSDOptions options;options.mode=sl::DLSSMode::eMaxQuality;options.outputWidth=w;options.outputHeight=h;sl::DLSSDOptimalSettings size;
        if(i.check(i.rr_size(options,size),"DLSS RR quality size"))return {size.optimalRenderWidth,size.optimalRenderHeight};i.rr_failed=true;
    }else if(mode==DxrReconstruction::NrdDlss) {
        sl::DLSSOptions options;options.mode=sl::DLSSMode::eMaxQuality;options.outputWidth=w;options.outputHeight=h;sl::DLSSOptimalSettings size;
        if(i.check(i.sr_size(options,size),"DLSS SR quality size"))return {size.optimalRenderWidth,size.optimalRenderHeight};i.sr_failed=true;
    }
    return {std::max(1u,(w*2)/3),std::max(1u,(h*2)/3)};
}
bool DxrStreamline::evaluate(D3d12Context& context,DxrReconstruction mode,const DxrFrameConstants& g,const DxrDlssInputs& input,const D3d12ResourcePtr& output) {
    auto& i=*impl_;const bool rr=mode==DxrReconstruction::DlssRr;if(!i.initialized || !i.presentation)return false;
    const auto feature=rr?sl::kFeatureDLSS_RR:sl::kFeatureDLSS;
    const Mat4 view=view_matrix(g.eye,g.forward,g.right,g.up),projection=projection_matrix(g.right,g.up);
    const Mat4 previous=view_matrix(g.previous_eye,g.previous_forward,g.previous_right,g.previous_up);
    const Mat4 previous_projection=projection_matrix(g.previous_right,g.previous_up);
    const Mat4 reprojection=previous_projection*previous*view.inverse()*projection.inverse();
    auto run=[&]()->bool {
        if(rr) {
            sl::DLSSOptions disabled;disabled.mode=sl::DLSSMode::eOff;disabled.outputWidth=g.size.z;disabled.outputHeight=g.size.w;
            if(i.sr_options)i.sr_options(i.viewport,disabled);
            sl::DLSSDOptions options;options.mode=sl::DLSSMode::eMaxQuality;options.outputWidth=g.size.z;options.outputHeight=g.size.w;
            options.normalRoughnessMode=sl::DLSSDNormalRoughnessMode::ePacked;options.qualityPreset=sl::DLSSDPreset::ePresetF;
            options.worldToCameraView=sl_matrix(view);options.cameraViewToWorld=sl_matrix(view.inverse());
            if(!i.check(i.rr_options(i.viewport,options),"DLSS RR options"))return false;
        }else {
            sl::DLSSDOptions disabled;disabled.mode=sl::DLSSMode::eOff;disabled.outputWidth=g.size.z;disabled.outputHeight=g.size.w;
            if(i.rr_options)i.rr_options(i.viewport,disabled);
            sl::DLSSOptions options;options.mode=sl::DLSSMode::eMaxQuality;options.outputWidth=g.size.z;options.outputHeight=g.size.w;options.qualityPreset=sl::DLSSPreset::ePresetK;
            if(!i.check(i.sr_options(i.viewport,options),"DLSS SR options"))return false;
        }
        sl::FrameToken* frame=nullptr;if(!i.check(i.frame_token(frame,&g.frame.x),"DLSS frame token"))return false;
        sl::Constants c;c.cameraViewToClip=sl_matrix(projection);c.clipToCameraView=sl_matrix(projection.inverse());c.clipToLensClip=sl_matrix(Mat4::Identity());
        c.clipToPrevClip=sl_matrix(reprojection);c.prevClipToClip=sl_matrix(reprojection.inverse());
        c.jitterOffset={g.jitter.x,g.jitter.y};c.mvecScale={1.f/g.size.x,1.f/g.size.y};c.cameraPinholeOffset={0,0};
        c.cameraPos={g.eye.x,g.eye.y,g.eye.z};c.cameraFwd={g.forward.x,g.forward.y,g.forward.z};c.cameraUp={g.up.x,g.up.y,g.up.z};c.cameraRight={g.right.x,g.right.y,g.right.z};
        c.cameraNear=.05f;c.cameraFar=1e6f;c.cameraFOV=2*std::atan(g.up.w*.5f);c.cameraAspectRatio=g.right.w/g.up.w;
        c.depthInverted=sl::Boolean::eFalse;c.cameraMotionIncluded=sl::Boolean::eTrue;c.motionVectors3D=sl::Boolean::eFalse;c.reset=g.frame.w?sl::Boolean::eTrue:sl::Boolean::eFalse;
        if(!i.check(i.constants(c,*frame,i.viewport),"DLSS constants"))return false;
        std::array<sl::Resource,9> resources;std::array<sl::ResourceTag,9> tags;
        const D3d12ResourcePtr pointers[]={input.color,output,input.depth,input.motion,input.normal_roughness,input.diffuse_albedo,input.specular_albedo,input.specular_distance,input.specular_motion};
        const sl::BufferType types[]={sl::kBufferTypeScalingInputColor,sl::kBufferTypeScalingOutputColor,sl::kBufferTypeDepth,sl::kBufferTypeMotionVectors,sl::kBufferTypeNormalRoughness,sl::kBufferTypeAlbedo,sl::kBufferTypeSpecularAlbedo,sl::kBufferTypeSpecularHitDistance,sl::kBufferTypeSpecularMotionVectors};
        const unsigned count=rr?9:4;
        for(unsigned j=0;j<count;++j) {
            context.transition(pointers[j],j==1?D3D12_RESOURCE_STATE_UNORDERED_ACCESS:D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            resources[j]=sl::Resource(sl::ResourceType::eTex2d,pointers[j]->resource.Get(),UINT(pointers[j]->state));
            const sl::Extent extent{0,0,j==1?g.size.z:g.size.x,j==1?g.size.w:g.size.y};
            tags[j]=sl::ResourceTag(&resources[j],types[j],sl::ResourceLifecycle::eValidUntilEvaluate,&extent);
        }
        if(!i.check(i.tags(*frame,i.viewport,tags.data(),count,context.commands()),"DLSS resource tags"))return false;
        const sl::BaseStructure* inputs[]={&i.viewport};if(!i.check(i.evaluate(feature,*frame,inputs,1,context.commands()),"DLSS evaluation"))return false;
        if(rr){sl::DLSSDState state;if(i.rr_state(i.viewport,state)==sl::Result::eOk)i.memory=state.estimatedVRAMUsageInBytes;}
        else{sl::DLSSState state;if(i.sr_state(i.viewport,state)==sl::Result::eOk)i.memory=state.estimatedVRAMUsageInBytes;}
        return true;
    };
    bool ok=run();
    if(ok)try{i.inspect_ngx();if(!i.pinned){ok=false;i.reason="Loaded DLSS runtime differs from pinned SDK; using NRD + TAAU";}}catch(const std::exception& error){ok=false;i.pinned=false;i.reason=error.what();}
    if(!ok){if(rr)i.rr_failed=true;else i.sr_failed=true;}
    // Streamline does not preserve root signatures, descriptor heaps, or PSOs.
    ID3D12DescriptorHeap* heaps[]={context.descriptors()->heap.Get(),context.samplers()};context.commands()->SetDescriptorHeaps(2,heaps);return ok;
}
const std::string& DxrStreamline::reason() const{return impl_->reason;}
std::uint64_t DxrStreamline::allocated_bytes() const{return impl_->memory;}
const std::vector<DxrRuntimeModule>& DxrStreamline::runtime_modules() const{return impl_->modules;}
bool DxrStreamline::runtime_pinned() const{return impl_->pinned;}
}
