#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "render/realtime/optix_backend.h"
#include "render/realtime/optix_denoiser.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <optix.h>
#include <optix_function_table_definition.h>
#include <optix_stubs.h>
#include <optix_stack_size.h>
#include <algorithm>
#include <array>
#include <climits>
#include <cstring>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <utility>
#include <vector>

namespace renderer {
namespace {
#include "rtrt_optix_ptx.h"
void cuda_check(cudaError_t status,const char* operation) {
    if(status!=cudaSuccess)throw std::runtime_error(std::string(operation)+": "+cudaGetErrorString(status));
}
void optix_check(OptixResult status,const char* operation,const char* log="") {
    // The function table is unavailable if optixInit itself failed.
    if(status!=OPTIX_SUCCESS)throw std::runtime_error(std::string(operation)+": OptiX error "+std::to_string(int(status))+" "+log);
}
void initialize_optix_driver() {
    static std::once_flag once;
    static OptixResult initialized=OPTIX_ERROR_UNKNOWN;
    std::call_once(once,[]{initialized=optixInit();});
    optix_check(initialized,"initialize OptiX driver");
}
struct Buffer {
    CUdeviceptr pointer=0;std::size_t bytes=0;
    Buffer()=default;Buffer(const Buffer&)=delete;Buffer& operator=(const Buffer&)=delete;
    Buffer(Buffer&& other) noexcept:pointer(std::exchange(other.pointer,0)),bytes(std::exchange(other.bytes,0)){}
    ~Buffer(){if(pointer)cudaFree(reinterpret_cast<void*>(pointer));}
    void resize(std::size_t size) {
        if(bytes==size)return;
        if(pointer)cuda_check(cudaFree(reinterpret_cast<void*>(pointer)),"free OptiX buffer");
        pointer=0;bytes=0;
        if(size)cuda_check(cudaMalloc(reinterpret_cast<void**>(&pointer),size),"allocate OptiX buffer");
        bytes=size;
    }
    void upload(const void* data,std::size_t size,cudaStream_t stream) {
        resize(size);if(size)cuda_check(cudaMemcpyAsync(reinterpret_cast<void*>(pointer),data,size,cudaMemcpyHostToDevice,stream),"upload OptiX data");
    }
};
struct alignas(OPTIX_SBT_RECORD_ALIGNMENT) Record {char header[OPTIX_SBT_RECORD_HEADER_SIZE];};
struct ParameterUpload {
    void* host=nullptr;std::size_t bytes=0;cudaEvent_t copied=nullptr;bool pending=false;
    ~ParameterUpload(){if(copied)cudaEventDestroy(copied);if(host)cudaFreeHost(host);}
    void stage(const void* source,std::size_t size,CUdeviceptr destination,cudaStream_t stream) {
        // Keep each pinned source alive until its queued copy completes, even
        // when callers submit several frames without waiting for the GPU.
        if(pending)cuda_check(cudaEventSynchronize(copied),"reuse OptiX parameter upload");
        if(bytes!=size) {
            if(host)cuda_check(cudaFreeHost(host),"free OptiX upload staging");
            host=nullptr;bytes=0;
            cuda_check(cudaHostAlloc(&host,size,cudaHostAllocDefault),"allocate OptiX upload staging");bytes=size;
        }
        if(!copied)cuda_check(cudaEventCreateWithFlags(&copied,cudaEventDisableTiming),"create OptiX upload event");
        std::memcpy(host,source,size);
        cuda_check(cudaMemcpyAsync(reinterpret_cast<void*>(destination),host,size,cudaMemcpyHostToDevice,stream),"upload OptiX launch parameters");
        cuda_check(cudaEventRecord(copied,stream),"record OptiX parameter upload");pending=true;
    }
};
}

class OptixRealtimeBackend::Impl {
public:
    explicit Impl(CudaDeviceContext device):device_(device) {}
    ~Impl() {
        device_.activate();cudaDeviceSynchronize();
        if(pipeline_)optixPipelineDestroy(pipeline_);
        for(auto group:groups_)if(group)optixProgramGroupDestroy(group);
        if(module_)optixModuleDestroy(module_);
        if(context_)optixDeviceContextDestroy(context_);
    }
    bool initialize() {
        if(failed_)return false;
        if(pipeline_)return true;
        try {
            device_.activate();
            initialize_optix_driver();
            OptixDeviceContextOptions context_options{};
            const bool validation=std::getenv("RTRT_OPTIX_VALIDATE")!=nullptr;
            if(validation)context_options.validationMode=OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
            optix_check(optixDeviceContextCreate(nullptr,&context_options,&context_),"create OptiX context");
            OptixPipelineCompileOptions compile{};
            compile.traversableGraphFlags=OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
            compile.numPayloadValues=2;compile.numAttributeValues=2;
            compile.pipelineLaunchParamsVariableName="rt_optix_params";
            compile.usesPrimitiveTypeFlags=OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
            if(validation)compile.exceptionFlags=OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW|OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
            OptixModuleCompileOptions module_options{};
            module_options.optLevel=validation?OPTIX_COMPILE_OPTIMIZATION_LEVEL_0:OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
            // OptiX requires full debug information for useful sanitizer device
            // backtraces. Keep it opt-in so production shader code is unchanged.
            module_options.debugLevel=validation?OPTIX_COMPILE_DEBUG_LEVEL_FULL:OPTIX_COMPILE_DEBUG_LEVEL_NONE;
            std::array<char,8192> log{};std::size_t log_size=log.size();
            auto result=optixModuleCreate(context_,&module_options,&compile,kRtrtOptixPtx,sizeof(kRtrtOptixPtx)-1,log.data(),&log_size,&module_);
            optix_check(result,"compile RTRT OptiX programs",log.data());
            std::array<OptixProgramGroupDesc,5> descriptors{};
            const char* raygens[]={"__raygen__primary","__raygen__lighting","__raygen__native_optics"};
            for(int i=0;i<3;++i) {
                descriptors[i].kind=OPTIX_PROGRAM_GROUP_KIND_RAYGEN;descriptors[i].raygen.module=module_;
                descriptors[i].raygen.entryFunctionName=raygens[i];
            }
            descriptors[3].kind=OPTIX_PROGRAM_GROUP_KIND_MISS;
            descriptors[3].miss.module=module_;descriptors[3].miss.entryFunctionName="__miss__scene";
            descriptors[4].kind=OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
            descriptors[4].hitgroup.moduleCH=module_;descriptors[4].hitgroup.entryFunctionNameCH="__closesthit__scene";
            descriptors[4].hitgroup.moduleAH=module_;descriptors[4].hitgroup.entryFunctionNameAH="__anyhit__scene";
            OptixProgramGroupOptions group_options{};log_size=log.size();
            result=optixProgramGroupCreate(context_,descriptors.data(),unsigned(descriptors.size()),&group_options,log.data(),&log_size,groups_.data());
            optix_check(result,"create OptiX programs",log.data());
            OptixPipelineLinkOptions link{};link.maxTraceDepth=1;log_size=log.size();
            result=optixPipelineCreate(context_,&compile,&link,groups_.data(),unsigned(groups_.size()),log.data(),&log_size,&pipeline_);
            optix_check(result,"link RTRT OptiX pipeline",log.data());
            OptixStackSizes sizes{};for(auto group:groups_)optix_check(optixUtilAccumulateStackSizes(group,&sizes,pipeline_),"query OptiX stack");
            unsigned traversal=0,state=0,continuation=0;
            optix_check(optixUtilComputeStackSizes(&sizes,1,0,0,&traversal,&state,&continuation),"compute OptiX stack");
            optix_check(optixPipelineSetStackSize(pipeline_,traversal,state,continuation,2),"configure OptiX stack");
            for(int i=0;i<7;++i)optix_check(optixSbtRecordPackHeader(groups_[std::min(i,4)],&records_host_[i]),"pack OptiX shader record");
            records_.upload(records_host_.data(),sizeof(records_host_),nullptr);
            cuda_check(cudaStreamSynchronize(nullptr),"finish OptiX program records");
            reason_.clear();return true;
        } catch(const std::exception& error) {reason_=error.what();failed_=true;return false;}
    }
    bool sync(const RenderSceneSnapshot& scene,cudaStream_t stream) {
        for(const auto& asset:scene.assets)if(asset.local_scene && !asset.local_scene->spheres.empty()) {
            reason_="Analytic sphere assets use CUDA traversal";return false;
        }
        if(!initialize())return false;
        device_.activate();
        bool rebuild=source_!=scene.source_id || assets_.size()!=scene.assets.size() ||
            materials_!=scene.revisions.materials || bindings_!=scene.revisions.material_bindings || topology_!=scene.revisions.topology;
        if(!rebuild)for(std::size_t i=0;i<assets_.size();++i) {
            const auto& a=scene.assets[i];
            if(assets_[i].source!=a.local_scene || assets_[i].id!=a.asset_id || assets_[i].revision!=a.geometry_revision){rebuild=true;break;}
        }
        if(rebuild) {
            cuda_check(cudaStreamSynchronize(stream),"rebuild OptiX scene");
            assets_.clear();assets_.resize(scene.assets.size());handle_=0;
            for(std::size_t i=0;i<assets_.size();++i) {
                auto& asset=assets_[i];const auto& input=scene.assets[i];
                asset.source=input.local_scene;asset.id=input.asset_id;asset.revision=input.geometry_revision;
                if(!input.local_scene || input.local_scene->triangles.empty())continue;
                const auto& triangles=input.local_scene->triangles;
                if(triangles.size()>std::size_t(UINT_MAX/3))throw std::runtime_error("OptiX mesh exceeds vertex limits");
                std::vector<float> positions;positions.reserve(triangles.size()*9);
                for(const auto& triangle:triangles)for(int v=0;v<3;++v) {
                    const auto& p=triangle.vertex(v).position;positions.insert(positions.end(),{p.x(),p.y(),p.z()});
                }
                asset.vertices.upload(positions.data(),positions.size()*sizeof(float),stream);
                // 0: opaque single sided, 1: opaque double sided, 2: material
                // acceptance shader. Shared meshes with different instance
                // bindings conservatively retain that shader when necessary.
                std::vector<unsigned char> geometry_class(triangles.size(),3);
                for(const auto& instance:scene.instances)if(instance.asset_index==int(i)) {
                    for(std::size_t t=0;t<triangles.size();++t) {
                        unsigned char kind=1;
                        if(t<input.triangle_material_slots.size()) {
                            const auto slot=input.triangle_material_slots[t];
                            if(slot.has_value() && slot.value()<instance.materials.size()) {
                                const auto& m=instance.materials[slot.value()];
                                const bool opaque=m.type==MaterialType::Pbr?m.alpha_mode==AlphaMode::Opaque:
                                    m.opacity_texture_id<0 && m.opacity>=1;
                                kind=opaque?(m.two_sided?1:0):2;
                            }
                        }
                        auto& previous=geometry_class[t];
                        previous=previous==3?kind:(previous==kind?kind:2);
                    }
                }
                for(auto& kind:geometry_class)if(kind==3)kind=1;
                asset.sbt_indices.upload(geometry_class.data(),geometry_class.size(),stream);
                OptixBuildInput build{};build.type=OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
                const unsigned flags[3]={OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
                    OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT|OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING,
                    OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING};
                build.triangleArray.vertexBuffers=&asset.vertices.pointer;
                build.triangleArray.numVertices=unsigned(triangles.size()*3);
                build.triangleArray.vertexFormat=OPTIX_VERTEX_FORMAT_FLOAT3;
                build.triangleArray.vertexStrideInBytes=3*sizeof(float);
                build.triangleArray.flags=flags;build.triangleArray.numSbtRecords=3;
                build.triangleArray.sbtIndexOffsetBuffer=asset.sbt_indices.pointer;
                build.triangleArray.sbtIndexOffsetSizeInBytes=1;
                OptixAccelBuildOptions options{};options.buildFlags=OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
                options.operation=OPTIX_BUILD_OPERATION_BUILD;
                build_acceleration(build,options,asset.gas,asset.handle,stream);
                // Acceleration build is asynchronous; retain the upload source
                // through completion, then release the scratch vertex buffer.
                cuda_check(cudaStreamSynchronize(stream),"finish OptiX mesh build");asset.vertices.resize(0);asset.sbt_indices.resize(0);
            }
        }
        const bool transform_changed=rebuild || transforms_!=scene.revisions.transforms || geometry_!=scene.revisions.geometry;
        if(transform_changed) {
            instances_host_.clear();
            for(std::size_t i=0;i<scene.instances.size();++i) {
                const auto& instance=scene.instances[i];
                if(instance.asset_index<0 || std::size_t(instance.asset_index)>=assets_.size() || !assets_[instance.asset_index].handle)continue;
                OptixInstance out{};
                for(int row=0;row<3;++row)for(int col=0;col<4;++col)out.transform[row*4+col]=instance.object_to_world(row,col);
                out.instanceId=unsigned(i);out.visibilityMask=255;
                if(instance.object_to_world.topLeftCorner<3,3>().determinant()<0)
                    out.flags=OPTIX_INSTANCE_FLAG_FLIP_TRIANGLE_FACING;
                out.traversableHandle=assets_[instance.asset_index].handle;instances_host_.push_back(out);
            }
            const bool update=!rebuild && handle_ && instances_.bytes==instances_host_.size()*sizeof(OptixInstance);
            instances_.upload(instances_host_.data(),instances_host_.size()*sizeof(OptixInstance),stream);
            if(instances_host_.empty())handle_=0;
            else {
                OptixBuildInput input{};input.type=OPTIX_BUILD_INPUT_TYPE_INSTANCES;
                input.instanceArray.instances=instances_.pointer;input.instanceArray.numInstances=unsigned(instances_host_.size());
                OptixAccelBuildOptions options{};options.buildFlags=OPTIX_BUILD_FLAG_PREFER_FAST_TRACE|OPTIX_BUILD_FLAG_ALLOW_UPDATE;
                options.operation=update?OPTIX_BUILD_OPERATION_UPDATE:OPTIX_BUILD_OPERATION_BUILD;
                build_acceleration(input,options,tlas_,handle_,stream);
            }
        }
        source_=scene.source_id;transforms_=scene.revisions.transforms;geometry_=scene.revisions.geometry;
        materials_=scene.revisions.materials;bindings_=scene.revisions.material_bindings;topology_=scene.revisions.topology;
        reason_.clear();return true;
    }
    void build_acceleration(const OptixBuildInput& input,const OptixAccelBuildOptions& options,Buffer& output,OptixTraversableHandle& handle,cudaStream_t stream) {
        OptixAccelBufferSizes sizes{};optix_check(optixAccelComputeMemoryUsage(context_,&options,&input,1,&sizes),"size OptiX acceleration");
        scratch_.resize(options.operation==OPTIX_BUILD_OPERATION_UPDATE?sizes.tempUpdateSizeInBytes:sizes.tempSizeInBytes);
        if(options.operation!=OPTIX_BUILD_OPERATION_UPDATE)output.resize(sizes.outputSizeInBytes);
        optix_check(optixAccelBuild(context_,stream,&options,&input,1,scratch_.pointer,scratch_.bytes,output.pointer,output.bytes,&handle,nullptr,0),"build OptiX acceleration");
    }
    void launch(OptixRealtimePass pass,const void* parameters,std::size_t bytes,int width,int height,cudaStream_t stream) {
        parameters_.resize(bytes);
        uploads_[upload_index_++%uploads_.size()].stage(parameters,bytes,parameters_.pointer,stream);
        OptixShaderBindingTable sbt{};
        sbt.raygenRecord=records_.pointer+static_cast<unsigned>(pass)*sizeof(Record);
        sbt.missRecordBase=records_.pointer+3*sizeof(Record);sbt.missRecordStrideInBytes=sizeof(Record);sbt.missRecordCount=1;
        sbt.hitgroupRecordBase=records_.pointer+4*sizeof(Record);sbt.hitgroupRecordStrideInBytes=sizeof(Record);sbt.hitgroupRecordCount=3;
        optix_check(optixLaunch(pipeline_,stream,parameters_.pointer,bytes,&sbt,unsigned(width),unsigned(height),1),"launch RTRT OptiX");
    }
    std::uint64_t resident_bytes() const {
        std::uint64_t total=tlas_.bytes+scratch_.bytes+instances_.bytes+records_.bytes+parameters_.bytes;
        for(const auto& asset:assets_)total+=asset.gas.bytes+asset.vertices.bytes;return total;
    }
    struct Asset {Buffer vertices,gas,sbt_indices;OptixTraversableHandle handle=0;std::uint64_t id=0,revision=0;std::shared_ptr<const Scene> source;};
    CudaDeviceContext device_;OptixDeviceContext context_=nullptr;OptixModule module_=nullptr;OptixPipeline pipeline_=nullptr;
    std::array<OptixProgramGroup,5> groups_{};std::array<Record,7> records_host_{};
    Buffer records_,parameters_,scratch_,instances_,tlas_;std::vector<Asset> assets_;std::vector<OptixInstance> instances_host_;
    std::array<ParameterUpload,8> uploads_{};std::size_t upload_index_=0;
    OptixTraversableHandle handle_=0;std::uint64_t source_=0,transforms_=0,geometry_=0;
    std::uint64_t materials_=0,bindings_=0,topology_=0;
    bool failed_=false;std::string reason_;
};
OptixRealtimeBackend::OptixRealtimeBackend(CudaDeviceContext device):impl_(std::make_unique<Impl>(device)){}
OptixRealtimeBackend::~OptixRealtimeBackend()=default;
bool OptixRealtimeBackend::sync(const RenderSceneSnapshot& scene,CudaStreamHandle stream){return impl_->sync(scene,reinterpret_cast<cudaStream_t>(stream));}
void OptixRealtimeBackend::launch(OptixRealtimePass pass,const void* p,std::size_t bytes,int w,int h,CudaStreamHandle stream){impl_->launch(pass,p,bytes,w,h,reinterpret_cast<cudaStream_t>(stream));}
std::uint64_t OptixRealtimeBackend::traversable() const{return impl_->handle_;}
std::uint64_t OptixRealtimeBackend::resident_bytes() const{return impl_->resident_bytes();}
const std::string& OptixRealtimeBackend::reason() const{return impl_->reason_;}

class OptixRealtimeDenoiser::Impl {
public:
    explicit Impl(CudaDeviceContext device):device_(device) {}
    ~Impl() {
        device_.activate();
        // The renderer drains its stream before destruction. No denoiser work
        // may outlive the model or the context that owns it.
        if(denoiser_)optixDenoiserDestroy(denoiser_);
        if(context_)optixDeviceContextDestroy(context_);
    }
    OptixImage2D image(CUdeviceptr data,OptixPixelFormat format=OPTIX_PIXEL_FORMAT_FLOAT3,unsigned stride=3*sizeof(float)) const {
        return {data,unsigned(width_),unsigned(height_),unsigned(width_)*stride,stride,format};
    }
    bool prepare(int width,int height,bool temporal,cudaStream_t stream,bool aovs) {
        if(failed_)return false;
        if(denoiser_ && width_==width && height_==height && temporal_==temporal && aovs_==aovs)return true;
        try {
            device_.activate();
            cuda_check(cudaStreamSynchronize(stream),"resize OptiX denoiser");
            if(denoiser_) {
                optix_check(optixDenoiserDestroy(denoiser_),"destroy OptiX denoiser");denoiser_=nullptr;
            }
            initialize_optix_driver();
            if(!context_) {
                OptixDeviceContextOptions options{};
                if(std::getenv("RTRT_OPTIX_VALIDATE"))options.validationMode=OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
                optix_check(optixDeviceContextCreate(nullptr,&options,&context_),"create OptiX denoiser context");
            }
            // OptiX 9.1 deprecates HDR/TEMPORAL in favor of these kernel-prediction models.
            OptixDenoiserOptions options{};options.guideAlbedo=1;options.guideNormal=1;
            optix_check(optixDenoiserCreate(context_,temporal?OPTIX_DENOISER_MODEL_KIND_TEMPORAL_AOV:
                OPTIX_DENOISER_MODEL_KIND_AOV,&options,&denoiser_),"create OptiX neural denoiser");
            width_=width;height_=height;temporal_=temporal;aovs_=aovs;valid_=false;index_=0;
            OptixDenoiserSizes sizes{};
            optix_check(optixDenoiserComputeMemoryResources(denoiser_,unsigned(width),unsigned(height),&sizes),"size OptiX denoiser");
            state_.resize(sizes.stateSizeInBytes);
            scratch_.resize(std::max(sizes.withoutOverlapScratchSizeInBytes,sizes.computeAverageColorSizeInBytes));
            const std::size_t pixels=std::size_t(width)*height;
            albedo_.resize(pixels*3*sizeof(float));normal_.resize(pixels*3*sizeof(float));
            flow_.resize(temporal?pixels*2*sizeof(float):0);trust_.resize(temporal?pixels*sizeof(float):0);
            average_.resize(3*sizeof(float));
            aov_input_.resize(aovs?pixels*12*sizeof(float):0);
            guide_stride_=unsigned(sizes.internalGuideLayerPixelSizeInBytes);
            for(int i=0;i<2;++i) {
                color_[i].resize((temporal || i==0)?pixels*3*sizeof(float)*(aovs?5:1):0);
                internal_[i].resize(temporal?pixels*guide_stride_:0);
            }
            optix_check(optixDenoiserSetup(denoiser_,stream,unsigned(width),unsigned(height),
                state_.pointer,state_.bytes,scratch_.pointer,scratch_.bytes),"set up OptiX denoiser");
            ++allocation_generation_;reason_.clear();return true;
        } catch(const std::exception& error) {
            const std::string message=error.what();release(stream);
            reason_=message;failed_=true;return false;
        }
    }
    bool invoke(const void* input,bool use_history,cudaStream_t stream) {
        if(failed_ || !denoiser_)return false;
        try {
            device_.activate();
            const int write=temporal_?1-index_:0;
            const bool previous=temporal_ && valid_ && use_history;
            OptixDenoiserGuideLayer guides{};
            guides.albedo=image(albedo_.pointer);guides.normal=image(normal_.pointer);
            if(temporal_) {
                guides.flow=image(flow_.pointer,OPTIX_PIXEL_FORMAT_FLOAT2,2*sizeof(float));
                guides.flowTrustworthiness=image(trust_.pointer,OPTIX_PIXEL_FORMAT_FLOAT1,sizeof(float));
                guides.previousOutputInternalGuideLayer=image(internal_[index_].pointer,OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER,guide_stride_);
                guides.outputInternalGuideLayer=image(internal_[write].pointer,OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER,guide_stride_);
                if(!previous)cuda_check(cudaMemsetAsync(reinterpret_cast<void*>(internal_[index_].pointer),0,
                    internal_[index_].bytes,stream),"reset OptiX internal history");
            }
            OptixDenoiserLayer layer{};layer.type=OPTIX_DENOISER_AOV_TYPE_BEAUTY;
            layer.input=image(reinterpret_cast<CUdeviceptr>(input));layer.output=image(color_[write].pointer);
            if(temporal_)layer.previousOutput=previous?image(color_[index_].pointer):layer.input;
            std::array<OptixDenoiserLayer,5> layers{};layers[0]=layer;
            if(aovs_) {
                const OptixDenoiserAOVType types[4]={OPTIX_DENOISER_AOV_TYPE_DIFFUSE,OPTIX_DENOISER_AOV_TYPE_DIFFUSE,
                    OPTIX_DENOISER_AOV_TYPE_REFLECTION,OPTIX_DENOISER_AOV_TYPE_REFRACTION};
                const std::size_t bytes=std::size_t(width_)*height_*3*sizeof(float);
                for(int i=0;i<4;++i) {
                    layers[i+1].input=image(aov_input_.pointer+i*bytes);
                    layers[i+1].output=image(color_[write].pointer+(i+1)*bytes);
                    layers[i+1].type=types[i];
                    if(temporal_)layers[i+1].previousOutput=previous?image(color_[index_].pointer+(i+1)*bytes):layers[i+1].input;
                }
            }
            OptixDenoiserParams params{};params.hdrAverageColor=average_.pointer;
            params.temporalModeUsePreviousLayers=previous?1:0;
            optix_check(optixDenoiserComputeAverageColor(denoiser_,stream,&layer.input,average_.pointer,
                scratch_.pointer,scratch_.bytes),"compute OptiX HDR exposure");
            optix_check(optixDenoiserInvoke(denoiser_,stream,&params,state_.pointer,state_.bytes,&guides,layers.data(),aovs_?5:1,0,0,
                scratch_.pointer,scratch_.bytes),"invoke OptiX neural denoiser");
            index_=write;valid_=true;return true;
        } catch(const std::exception& error) {
            const std::string message=error.what();release(stream);
            reason_=message;failed_=true;return false;
        }
    }
    void release(cudaStream_t stream) {
        if(denoiser_ || state_.bytes) {
            device_.activate();cuda_check(cudaStreamSynchronize(stream),"release OptiX denoiser");
            if(denoiser_) {optix_check(optixDenoiserDestroy(denoiser_),"destroy OptiX denoiser");denoiser_=nullptr;}
            for(auto* buffer:{&state_,&scratch_,&albedo_,&normal_,&flow_,&trust_,&average_,&aov_input_,
                &color_[0],&color_[1],&internal_[0],&internal_[1]})buffer->resize(0);
        }
        valid_=false;failed_=false;reason_.clear();
    }
    std::uint64_t resident_bytes() const {
        return state_.bytes+scratch_.bytes+albedo_.bytes+normal_.bytes+flow_.bytes+trust_.bytes+average_.bytes+aov_input_.bytes+
            color_[0].bytes+color_[1].bytes+internal_[0].bytes+internal_[1].bytes;
    }
    CudaDeviceContext device_;
    OptixDeviceContext context_=nullptr;
    OptixDenoiser denoiser_=nullptr;
    Buffer state_,scratch_,albedo_,normal_,flow_,trust_,average_,aov_input_;
    std::array<Buffer,2> color_,internal_;
    int width_=0,height_=0,index_=0;
    unsigned guide_stride_=0;
    std::uint64_t allocation_generation_=0;
    bool temporal_=false,aovs_=false,valid_=false,failed_=false;
    std::string reason_;
};
OptixRealtimeDenoiser::OptixRealtimeDenoiser(CudaDeviceContext device):impl_(std::make_unique<Impl>(device)){}
OptixRealtimeDenoiser::~OptixRealtimeDenoiser()=default;
bool OptixRealtimeDenoiser::prepare(int w,int h,bool temporal,CudaStreamHandle stream,bool aovs){return impl_->prepare(w,h,temporal,reinterpret_cast<cudaStream_t>(stream),aovs);}
bool OptixRealtimeDenoiser::invoke(const void* input,bool history,CudaStreamHandle stream){return impl_->invoke(input,history,reinterpret_cast<cudaStream_t>(stream));}
void OptixRealtimeDenoiser::release(CudaStreamHandle stream){impl_->release(reinterpret_cast<cudaStream_t>(stream));}
OptixDenoiserGuides OptixRealtimeDenoiser::guides() const {
    return {reinterpret_cast<float*>(impl_->albedo_.pointer),reinterpret_cast<float*>(impl_->normal_.pointer),
        reinterpret_cast<float*>(impl_->flow_.pointer),reinterpret_cast<float*>(impl_->trust_.pointer)};
}
const void* OptixRealtimeDenoiser::output() const{return reinterpret_cast<const void*>(impl_->color_[impl_->index_].pointer);}
void* OptixRealtimeDenoiser::aov_input() const{return reinterpret_cast<void*>(impl_->aov_input_.pointer);}
const void* OptixRealtimeDenoiser::aov_output() const {
    return reinterpret_cast<const void*>(impl_->color_[impl_->index_].pointer+std::size_t(impl_->width_)*impl_->height_*3*sizeof(float));
}
std::uint64_t OptixRealtimeDenoiser::resident_bytes() const{return impl_->resident_bytes();}
std::uint64_t OptixRealtimeDenoiser::allocation_generation() const{return impl_->allocation_generation_;}
const std::string& OptixRealtimeDenoiser::reason() const{return impl_->reason_;}
}
