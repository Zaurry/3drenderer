#ifndef NOMINMAX
#define NOMINMAX
#endif
#include "render/optix/optix_backend.h"
#include "render/optix/optix_denoiser.h"
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
#include <unordered_map>
#include <unordered_set>
#include "render/optix/optix_realtime_renderer.h"

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
    void reserve(std::size_t size) { if(size>bytes)resize(size); }
    void swap(Buffer& other) noexcept {std::swap(pointer,other.pointer);std::swap(bytes,other.bytes);}
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
        device_.activate();cudaStreamSynchronize(last_stream_);
        if(pipeline_)optixPipelineDestroy(pipeline_);
        for(auto group:groups_)if(group)optixProgramGroupDestroy(group);
        if(module_)optixModuleDestroy(module_);
        if(sphere_module_)optixModuleDestroy(sphere_module_);
        if(accel_start_)cudaEventDestroy(accel_start_);
        if(accel_end_)cudaEventDestroy(accel_end_);
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
            optix_check(optixDeviceContextGetProperty(context_,OPTIX_DEVICE_PROPERTY_RTCORE_VERSION,&rt_core_version_,sizeof(rt_core_version_)),"query RT Core capability");
            if(!rt_core_version_)throw std::runtime_error("RTRT requires an RT Core capable device");
            unsigned ser=0;
            optix_check(optixDeviceContextGetProperty(context_,OPTIX_DEVICE_PROPERTY_SHADER_EXECUTION_REORDERING,&ser,sizeof(ser)),"query SER capability");
            ser_supported_=(ser&OPTIX_DEVICE_PROPERTY_SHADER_EXECUTION_REORDERING_FLAG_STANDARD)!=0;
            cuda_check(cudaEventCreate(&accel_start_),"create acceleration timer");
            cuda_check(cudaEventCreate(&accel_end_),"create acceleration timer");
            OptixPipelineCompileOptions compile{};
            compile.traversableGraphFlags=OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
            compile.numPayloadValues=2;compile.numAttributeValues=2;
            compile.pipelineLaunchParamsVariableName="rt_optix_params";
            compile.usesPrimitiveTypeFlags=OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE|OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;
            if(validation)compile.exceptionFlags=OPTIX_EXCEPTION_FLAG_STACK_OVERFLOW|OPTIX_EXCEPTION_FLAG_TRACE_DEPTH;
            OptixModuleCompileOptions module_options{};
            module_options.optLevel=validation?OPTIX_COMPILE_OPTIMIZATION_LEVEL_0:OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
            // OptiX requires full debug information for useful sanitizer device
            // backtraces. Keep it opt-in so production shader code is unchanged.
            module_options.debugLevel=validation?OPTIX_COMPILE_DEBUG_LEVEL_FULL:OPTIX_COMPILE_DEBUG_LEVEL_NONE;
            std::array<char,8192> log{};std::size_t log_size=log.size();
            auto result=optixModuleCreate(context_,&module_options,&compile,reinterpret_cast<const char*>(kRtrtOptixPtx),sizeof(kRtrtOptixPtx)-1,log.data(),&log_size,&module_);
            optix_check(result,"compile RTRT OptiX programs",log.data());
            OptixBuiltinISOptions sphere_options{};
            sphere_options.builtinISModuleType=OPTIX_PRIMITIVE_TYPE_SPHERE;
            sphere_options.buildFlags=gas_flags;
            optix_check(optixBuiltinISModuleGet(context_,&module_options,&compile,&sphere_options,&sphere_module_),"create sphere intersection module");
            std::array<OptixProgramGroupDesc,9> descriptors{};
            const char* raygens[]={"__raygen__primary","__raygen__lighting","__raygen__native_optics"};
            for(int i=0;i<3;++i) {
                descriptors[i].kind=OPTIX_PROGRAM_GROUP_KIND_RAYGEN;descriptors[i].raygen.module=module_;
                descriptors[i].raygen.entryFunctionName=raygens[i];
            }
            for(int ray=0;ray<2;++ray) {
                descriptors[3+ray].kind=OPTIX_PROGRAM_GROUP_KIND_MISS;
                descriptors[3+ray].miss.module=module_;
                descriptors[3+ray].miss.entryFunctionName=ray?"__miss__occlusion":"__miss__radiance";
                for(int primitive=0;primitive<2;++primitive) {
                    auto& d=descriptors[5+2*primitive+ray];d.kind=OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
                    if(!ray) {d.hitgroup.moduleCH=module_;d.hitgroup.entryFunctionNameCH="__closesthit__scene";}
                    d.hitgroup.moduleAH=module_;d.hitgroup.entryFunctionNameAH=ray?"__anyhit__occlusion":"__anyhit__radiance";
                    if(primitive)d.hitgroup.moduleIS=sphere_module_;
                }
            }
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
            for(int i=0;i<5;++i)optix_check(optixSbtRecordPackHeader(groups_[i],&records_host_[i]),"pack OptiX shader record");
            for(int primitive=0;primitive<2;++primitive)for(int kind=0;kind<3;++kind)for(int ray=0;ray<2;++ray)
                optix_check(optixSbtRecordPackHeader(groups_[5+primitive*2+ray],&records_host_[5+primitive*6+kind*2+ray]),"pack OptiX hit record");
            records_.upload(records_host_.data(),sizeof(records_host_),nullptr);
            cuda_check(cudaStreamSynchronize(nullptr),"finish OptiX program records");
            reason_.clear();return true;
        } catch(const std::exception& error) {reason_=error.what();failed_=true;return false;}
    }
    static constexpr unsigned gas_flags=OPTIX_BUILD_FLAG_PREFER_FAST_TRACE|OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
    struct Geometry {
        Buffer vertices,classes,gas,compacted_size;
        OptixTraversableHandle handle=0;
        std::vector<float> vertices_host;
        std::vector<unsigned char> classes_host;
    };
    struct Asset {
        Geometry triangles,spheres;
        std::uint64_t revision=0;
        std::shared_ptr<const Scene> source;
    };
    std::vector<unsigned char> classify(const RenderSceneSnapshot& scene,std::size_t index,bool spheres) {
        const auto& input=scene.assets[index];
        const auto& slots=spheres?input.sphere_material_slots:input.triangle_material_slots;
        const auto count=spheres?input.local_scene->spheres.size():input.local_scene->triangles.size();
        std::vector<unsigned char> classes(count,3);
        for(const auto& instance:scene.instances)if(instance.asset_index==int(index)) {
            for(std::size_t t=0;t<count;++t) {
                unsigned char kind=1;
                if(t<slots.size() && slots[t].has_value() && slots[t].value()<instance.materials.size()) {
                    const auto& m=instance.materials[slots[t].value()];
                    const bool opaque=m.type==MaterialType::Pbr?m.alpha_mode==AlphaMode::Opaque:
                        m.opacity_texture_id<0 && m.opacity>=1;
                    kind=opaque?(m.two_sided?1:0):2;
                }
                auto& previous=classes[t];previous=previous==3?kind:(previous==kind?kind:2);
            }
        }
        for(auto& kind:classes)if(kind==3)kind=1;
        return classes;
    }
    void build_geometry(Geometry& geometry,const Scene& scene,bool spheres,cudaStream_t stream) {
        geometry.handle=0;
        const std::size_t count=spheres?scene.spheres.size():scene.triangles.size();
        if(!count) {geometry.gas.resize(0);return;}
        if(count>std::size_t(UINT_MAX/3))throw std::runtime_error("OptiX mesh exceeds primitive limits");
        auto& positions=geometry.vertices_host;positions.clear();positions.reserve(count*(spheres?4:9));
        if(spheres)for(const auto& sphere:scene.spheres) {
            const auto& p=sphere.center();positions.insert(positions.end(),{p.x(),p.y(),p.z(),sphere.radius()});
        } else for(const auto& triangle:scene.triangles)for(int v=0;v<3;++v) {
            const auto& p=triangle.vertex(v).position;positions.insert(positions.end(),{p.x(),p.y(),p.z()});
        }
        geometry.vertices.upload(positions.data(),positions.size()*sizeof(float),stream);
        geometry.classes.upload(geometry.classes_host.data(),geometry.classes_host.size(),stream);
        OptixBuildInput build{};
        const unsigned triangle_flags[3]={OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,
            OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT|OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING,
            OPTIX_GEOMETRY_FLAG_DISABLE_TRIANGLE_FACE_CULLING};
        const unsigned sphere_flags[3]={OPTIX_GEOMETRY_FLAG_NONE,OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT,OPTIX_GEOMETRY_FLAG_NONE};
        CUdeviceptr radii=geometry.vertices.pointer+3*sizeof(float);
        if(spheres) {
            build.type=OPTIX_BUILD_INPUT_TYPE_SPHERES;auto& a=build.sphereArray;
            a.vertexBuffers=&geometry.vertices.pointer;a.numVertices=unsigned(count);a.vertexStrideInBytes=4*sizeof(float);
            a.radiusBuffers=&radii;a.radiusStrideInBytes=4*sizeof(float);
            a.flags=sphere_flags;a.numSbtRecords=3;a.sbtIndexOffsetBuffer=geometry.classes.pointer;a.sbtIndexOffsetSizeInBytes=1;
        } else {
            build.type=OPTIX_BUILD_INPUT_TYPE_TRIANGLES;auto& a=build.triangleArray;
            a.vertexBuffers=&geometry.vertices.pointer;a.numVertices=unsigned(count*3);
            a.vertexFormat=OPTIX_VERTEX_FORMAT_FLOAT3;a.vertexStrideInBytes=3*sizeof(float);
            a.flags=triangle_flags;a.numSbtRecords=3;a.sbtIndexOffsetBuffer=geometry.classes.pointer;a.sbtIndexOffsetSizeInBytes=1;
        }
        OptixAccelBuildOptions options{};options.buildFlags=gas_flags;options.operation=OPTIX_BUILD_OPERATION_BUILD;
        geometry.compacted_size.resize(sizeof(std::uint64_t));
        OptixAccelEmitDesc emit{geometry.compacted_size.pointer,OPTIX_PROPERTY_TYPE_COMPACTED_SIZE};
        build_acceleration(build,options,geometry.gas,geometry.handle,stream,&emit);
        ++gas_builds_;
    }
    bool sync(const RenderSceneSnapshot& scene,cudaStream_t stream) {
        if(!initialize())return false;
        device_.activate();
        last_stream_=stream;
        update_timing();
        const bool different=source_!=scene.source_id;
        const bool classification_changed=different || materials_!=scene.revisions.materials ||
            bindings_!=scene.revisions.material_bindings || topology_!=scene.revisions.topology;
        if(different) {cuda_check(cudaStreamSynchronize(stream),"replace OptiX scene");assets_.clear();handle_=0;}
        std::unordered_set<std::uint64_t> live;
        struct Job {Geometry* geometry;const Scene* scene;bool spheres;};
        std::vector<Job> jobs;
        for(std::size_t i=0;i<scene.assets.size();++i) {
            const auto& input=scene.assets[i];
            if(!input.local_scene)throw std::runtime_error("OptiX asset has no geometry");
            live.insert(input.asset_id);
            auto& entry=assets_[input.asset_id];if(!entry)entry=std::make_unique<Asset>();
            auto& asset=*entry;
            const bool geometry_changed=asset.source!=input.local_scene || asset.revision!=input.geometry_revision;
            for(int sphere=0;sphere<2;++sphere) {
                auto& geometry=sphere?asset.spheres:asset.triangles;
                if(classification_changed || geometry_changed) {
                    auto classes=classify(scene,i,sphere!=0);
                    if(geometry_changed || classes!=geometry.classes_host) {
                        geometry.classes_host=std::move(classes);jobs.push_back({&geometry,input.local_scene.get(),sphere!=0});
                    }
                }
            }
            asset.source=input.local_scene;asset.revision=input.geometry_revision;
        }
        bool removed=false;for(const auto& [id,asset]:assets_)removed|=!live.contains(id);
        const bool changed=different || !jobs.empty() || removed || transforms_!=scene.revisions.transforms ||
            geometry_!=scene.revisions.geometry || topology_!=scene.revisions.topology;
        if(!changed) {materials_=scene.revisions.materials;bindings_=scene.revisions.material_bindings;return true;}
        // Geometry storage and compacted handles must outlive all launches using
        // them. Rigid transforms do not take this synchronization path.
        if(!jobs.empty() || removed)cuda_check(cudaStreamSynchronize(stream),"replace OptiX geometry");
        for(auto i=assets_.begin();i!=assets_.end();)if(!live.contains(i->first))i=assets_.erase(i);else ++i;
        const bool timing=!accel_pending_;
        if(timing)cuda_check(cudaEventRecord(accel_start_,stream),"start acceleration timer");
        for(auto& job:jobs)build_geometry(*job.geometry,*job.scene,job.spheres,stream);
        std::vector<std::uint64_t> compact_sizes(jobs.size());
        for(std::size_t i=0;i<jobs.size();++i)if(jobs[i].geometry->handle)
            cuda_check(cudaMemcpyAsync(&compact_sizes[i],reinterpret_cast<void*>(jobs[i].geometry->compacted_size.pointer),sizeof(std::uint64_t),cudaMemcpyDeviceToHost,stream),"query compacted GAS size");
        if(!jobs.empty())cuda_check(cudaStreamSynchronize(stream),"finish GAS build batch");
        std::vector<Buffer> retired;retired.reserve(jobs.size());
        for(std::size_t i=0;i<jobs.size();++i) {
            auto& g=*jobs[i].geometry;
            if(compact_sizes[i] && compact_sizes[i]<g.gas.bytes) {
                Buffer compact;compact.resize(std::size_t(compact_sizes[i]));
                optix_check(optixAccelCompact(context_,stream,g.handle,compact.pointer,compact.bytes,&g.handle),"compact OptiX GAS");
                g.gas.swap(compact);retired.push_back(std::move(compact));
            }
        }
        if(!jobs.empty()) {
            cuda_check(cudaStreamSynchronize(stream),"finish GAS compaction batch");
            for(auto& job:jobs) {auto& g=*job.geometry;g.vertices.resize(0);g.classes.resize(0);g.compacted_size.resize(0);g.vertices_host.clear();g.vertices_host.shrink_to_fit();}
        }
        instances_host_.clear();
        for(std::size_t i=0;i<scene.instances.size();++i) {
            const auto& instance=scene.instances[i];
            if(instance.asset_index<0 || std::size_t(instance.asset_index)>=scene.assets.size())throw std::runtime_error("OptiX instance references an invalid asset");
            const auto& asset=*assets_.at(scene.assets[instance.asset_index].asset_id);
            for(int sphere=0;sphere<2;++sphere) {
                const auto handle=sphere?asset.spheres.handle:asset.triangles.handle;if(!handle)continue;
                OptixInstance out{};
                for(int row=0;row<3;++row)for(int col=0;col<4;++col)out.transform[row*4+col]=instance.object_to_world(row,col);
                out.instanceId=unsigned(i);out.visibilityMask=255;out.sbtOffset=sphere?6:0;
                if(!sphere && instance.object_to_world.topLeftCorner<3,3>().determinant()<0)out.flags=OPTIX_INSTANCE_FLAG_FLIP_TRIANGLE_FACING;
                out.traversableHandle=handle;instances_host_.push_back(out);
            }
        }
        if(instances_host_.empty())handle_=0;
        else {
            const auto bytes=instances_host_.size()*sizeof(OptixInstance);
            const bool update=handle_ && jobs.empty() && !removed && instance_count_==instances_host_.size() && topology_==scene.revisions.topology;
            instances_.reserve(bytes);
            instance_uploads_[instance_upload_index_++%instance_uploads_.size()].stage(instances_host_.data(),bytes,instances_.pointer,stream);
            OptixBuildInput input{};input.type=OPTIX_BUILD_INPUT_TYPE_INSTANCES;
            input.instanceArray.instances=instances_.pointer;input.instanceArray.numInstances=unsigned(instances_host_.size());
            OptixAccelBuildOptions options{};options.buildFlags=OPTIX_BUILD_FLAG_PREFER_FAST_TRACE|OPTIX_BUILD_FLAG_ALLOW_UPDATE;
            options.operation=update?OPTIX_BUILD_OPERATION_UPDATE:OPTIX_BUILD_OPERATION_BUILD;
            build_acceleration(input,options,tlas_,handle_,stream);
            if(update)++ias_updates_;else ++ias_builds_;
        }
        instance_count_=instances_host_.size();
        if(timing) {cuda_check(cudaEventRecord(accel_end_,stream),"finish acceleration timer");accel_pending_=true;}
        source_=scene.source_id;transforms_=scene.revisions.transforms;geometry_=scene.revisions.geometry;
        materials_=scene.revisions.materials;bindings_=scene.revisions.material_bindings;topology_=scene.revisions.topology;
        reason_.clear();return true;
    }
    void update_timing() {
        if(!accel_pending_)return;
        const auto result=cudaEventQuery(accel_end_);
        if(result==cudaSuccess) {cuda_check(cudaEventElapsedTime(&acceleration_ms_,accel_start_,accel_end_),"read acceleration timer");accel_pending_=false;}
        else if(result!=cudaErrorNotReady)cuda_check(result,"query acceleration timer");
    }
    void build_acceleration(const OptixBuildInput& input,const OptixAccelBuildOptions& options,Buffer& output,OptixTraversableHandle& handle,cudaStream_t stream,const OptixAccelEmitDesc* emit=nullptr) {
        OptixAccelBufferSizes sizes{};optix_check(optixAccelComputeMemoryUsage(context_,&options,&input,1,&sizes),"size OptiX acceleration");
        scratch_.reserve(options.operation==OPTIX_BUILD_OPERATION_UPDATE?sizes.tempUpdateSizeInBytes:sizes.tempSizeInBytes);
        if(options.operation!=OPTIX_BUILD_OPERATION_UPDATE)output.reserve(sizes.outputSizeInBytes);
        optix_check(optixAccelBuild(context_,stream,&options,&input,1,scratch_.pointer,scratch_.bytes,output.pointer,output.bytes,&handle,emit,emit?1:0),"build OptiX acceleration");
    }
    void launch(OptixRealtimePass pass,const void* parameters,std::size_t bytes,int width,int height,cudaStream_t stream) {
        last_stream_=stream;
        parameters_.resize(bytes);
        uploads_[upload_index_++%uploads_.size()].stage(parameters,bytes,parameters_.pointer,stream);
        OptixShaderBindingTable sbt{};
        sbt.raygenRecord=records_.pointer+static_cast<unsigned>(pass)*sizeof(Record);
        sbt.missRecordBase=records_.pointer+3*sizeof(Record);sbt.missRecordStrideInBytes=sizeof(Record);sbt.missRecordCount=2;
        sbt.hitgroupRecordBase=records_.pointer+5*sizeof(Record);sbt.hitgroupRecordStrideInBytes=sizeof(Record);sbt.hitgroupRecordCount=12;
        optix_check(optixLaunch(pipeline_,stream,parameters_.pointer,bytes,&sbt,unsigned(width),unsigned(height),1),"launch RTRT OptiX");
    }
    std::uint64_t resident_bytes() const {
        std::uint64_t total=tlas_.bytes+scratch_.bytes+instances_.bytes+records_.bytes+parameters_.bytes;
        for(const auto& [id,asset]:assets_)for(const auto* g:{&asset->triangles,&asset->spheres})
            total+=g->gas.bytes+g->vertices.bytes+g->classes.bytes+g->compacted_size.bytes;
        return total;
    }
    CudaDeviceContext device_;OptixDeviceContext context_=nullptr;OptixModule module_=nullptr,sphere_module_=nullptr;OptixPipeline pipeline_=nullptr;
    cudaStream_t last_stream_=nullptr;
    std::array<OptixProgramGroup,9> groups_{};std::array<Record,17> records_host_{};
    Buffer records_,parameters_,scratch_,instances_,tlas_;
    std::unordered_map<std::uint64_t,std::unique_ptr<Asset>> assets_;std::vector<OptixInstance> instances_host_;
    std::array<ParameterUpload,8> uploads_{},instance_uploads_{};std::size_t upload_index_=0,instance_upload_index_=0,instance_count_=0;
    unsigned rt_core_version_=0;bool ser_supported_=false;
    std::uint64_t gas_builds_=0,ias_builds_=0,ias_updates_=0;
    cudaEvent_t accel_start_=nullptr,accel_end_=nullptr;bool accel_pending_=false;float acceleration_ms_=0;
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

unsigned OptixRealtimeBackend::rt_core_version() const{return impl_->rt_core_version_;}
bool OptixRealtimeBackend::ser_supported() const{return impl_->ser_supported_;}
std::uint64_t OptixRealtimeBackend::gas_builds() const{return impl_->gas_builds_;}
std::uint64_t OptixRealtimeBackend::ias_builds() const{return impl_->ias_builds_;}
std::uint64_t OptixRealtimeBackend::ias_updates() const{return impl_->ias_updates_;}
float OptixRealtimeBackend::acceleration_ms() const{impl_->update_timing();return impl_->acceleration_ms_;}

bool optix_realtime_available(int device,std::string* reason) {
    static std::mutex mutex;
    static std::unordered_map<int,std::string> results;
    std::lock_guard lock(mutex);
    const auto cached=results.find(device);
    if(cached!=results.end()){if(reason)*reason=cached->second;return cached->second.empty();}
    std::string detail;OptixDeviceContext context=nullptr;
    try {
        auto cuda=CudaDeviceContext::create(device);cuda.activate();initialize_optix_driver();
        OptixDeviceContextOptions options{};
        optix_check(optixDeviceContextCreate(nullptr,&options,&context),"probe OptiX context");
        unsigned rt=0;optix_check(optixDeviceContextGetProperty(context,OPTIX_DEVICE_PROPERTY_RTCORE_VERSION,&rt,sizeof(rt)),"probe RT Cores");
        if(!rt)detail="RTRT requires an RT Core capable device";
    } catch(const std::exception& e){detail=e.what();}
    if(context)optixDeviceContextDestroy(context);
    results.emplace(device,detail);if(reason)*reason=detail;return detail.empty();
}

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
