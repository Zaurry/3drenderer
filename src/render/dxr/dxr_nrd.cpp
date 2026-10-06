#include "render/dxr/dxr_nrd.h"
#include "core/math/types.h"
#include <NRD.h>
#include <cstring>
#include <stdexcept>

namespace renderer {
namespace {
void check_nrd(nrd::Result result,const char* label){if(result!=nrd::Result::SUCCESS)throw std::runtime_error(std::string(label)+": NRD error "+std::to_string(unsigned(result)));}
DXGI_FORMAT format(nrd::Format f) {
    constexpr DXGI_FORMAT formats[]={
        DXGI_FORMAT_R8_UNORM,DXGI_FORMAT_R8_SNORM,DXGI_FORMAT_R8_UINT,DXGI_FORMAT_R8_SINT,
        DXGI_FORMAT_R8G8_UNORM,DXGI_FORMAT_R8G8_SNORM,DXGI_FORMAT_R8G8_UINT,DXGI_FORMAT_R8G8_SINT,
        DXGI_FORMAT_R8G8B8A8_UNORM,DXGI_FORMAT_R8G8B8A8_SNORM,DXGI_FORMAT_R8G8B8A8_UINT,DXGI_FORMAT_R8G8B8A8_SINT,DXGI_FORMAT_R8G8B8A8_UNORM_SRGB,
        DXGI_FORMAT_R16_UNORM,DXGI_FORMAT_R16_SNORM,DXGI_FORMAT_R16_UINT,DXGI_FORMAT_R16_SINT,DXGI_FORMAT_R16_FLOAT,
        DXGI_FORMAT_R16G16_UNORM,DXGI_FORMAT_R16G16_SNORM,DXGI_FORMAT_R16G16_UINT,DXGI_FORMAT_R16G16_SINT,DXGI_FORMAT_R16G16_FLOAT,
        DXGI_FORMAT_R16G16B16A16_UNORM,DXGI_FORMAT_R16G16B16A16_SNORM,DXGI_FORMAT_R16G16B16A16_UINT,DXGI_FORMAT_R16G16B16A16_SINT,DXGI_FORMAT_R16G16B16A16_FLOAT,
        DXGI_FORMAT_R32_UINT,DXGI_FORMAT_R32_SINT,DXGI_FORMAT_R32_FLOAT,
        DXGI_FORMAT_R32G32_UINT,DXGI_FORMAT_R32G32_SINT,DXGI_FORMAT_R32G32_FLOAT,
        DXGI_FORMAT_R32G32B32_UINT,DXGI_FORMAT_R32G32B32_SINT,DXGI_FORMAT_R32G32B32_FLOAT,
        DXGI_FORMAT_R32G32B32A32_UINT,DXGI_FORMAT_R32G32B32A32_SINT,DXGI_FORMAT_R32G32B32A32_FLOAT,
        DXGI_FORMAT_R10G10B10A2_UNORM,DXGI_FORMAT_R10G10B10A2_UINT,DXGI_FORMAT_R11G11B10_FLOAT,DXGI_FORMAT_R9G9B9E5_SHAREDEXP};
    static_assert(std::size(formats)==std::size_t(nrd::Format::MAX_NUM));return formats[std::size_t(f)];
}
void camera_matrices(const DxrFloat4& eye,const DxrFloat4& f,const DxrFloat4& r,const DxrFloat4& u,float* view,float* projection) {
    Mat4 v=Mat4::Identity();v.row(0)=Vec4(r.x,r.y,r.z,-r.x*eye.x-r.y*eye.y-r.z*eye.z);
    v.row(1)=Vec4(u.x,u.y,u.z,-u.x*eye.x-u.y*eye.y-u.z*eye.z);
    v.row(2)=Vec4(-f.x,-f.y,-f.z,f.x*eye.x+f.y*eye.y+f.z*eye.z);
    Mat4 p=Mat4::Zero();p(0,0)=2/r.w;p(1,1)=2/u.w;p(2,2)=-1;p(2,3)=-.05f;p(3,2)=-1;
    std::memcpy(view,v.data(),sizeof(float)*16);std::memcpy(projection,p.data(),sizeof(float)*16);
}
}
struct DxrNrd::Impl {
    std::shared_ptr<D3d12Context> context;
    nrd::Instance* instance=nullptr;
    struct Pipeline {ComPtr<ID3D12RootSignature> root;ComPtr<ID3D12PipelineState> state;};
    std::vector<Pipeline> pipelines;
    std::vector<D3d12ResourcePtr> permanent,transient;
    D3d12ResourcePtr diffuse,specular,validation;
    std::array<std::shared_ptr<D3d12Descriptor>,3> tables;
    UINT table_stride=0,width=0,height=0;
    float previous_jitter[2]{};
    ~Impl(){if(instance)nrd::DestroyInstance(*instance);}
    explicit Impl(std::shared_ptr<D3d12Context> c):context(std::move(c)) {
        const nrd::DenoiserDesc denoiser{0,nrd::Denoiser::RELAX_DIFFUSE_SPECULAR};nrd::InstanceCreationDesc creation{};creation.denoisers=&denoiser;creation.denoisersNum=1;
        check_nrd(nrd::CreateInstance(creation,instance),"Create RELAX");
        const auto destroy=[](nrd::Instance* value){if(value)nrd::DestroyInstance(*value);};
        std::unique_ptr<nrd::Instance,decltype(destroy)> construction_guard(instance,destroy);
        const auto& description=*nrd::GetInstanceDesc(*instance);table_stride=description.descriptorPoolDesc.perSetTexturesMaxNum+description.descriptorPoolDesc.perSetStorageTexturesMaxNum;
        for(auto& table:tables)table=context->descriptor(table_stride*description.descriptorPoolDesc.setsMaxNum);
        pipelines.resize(description.pipelinesNum);
        for(UINT i=0;i<description.pipelinesNum;++i) {
            const auto& pipeline=description.pipelines[i];D3D12_DESCRIPTOR_RANGE ranges[2]{};UINT offset=0;
            for(UINT j=0;j<pipeline.resourceRangesNum;++j) {
                ranges[j].RangeType=pipeline.resourceRanges[j].descriptorType==nrd::DescriptorType::TEXTURE?D3D12_DESCRIPTOR_RANGE_TYPE_SRV:D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
                ranges[j].NumDescriptors=pipeline.resourceRanges[j].descriptorsNum;ranges[j].BaseShaderRegister=description.resourcesBaseRegisterIndex;
                ranges[j].RegisterSpace=description.resourcesSpaceIndex;ranges[j].OffsetInDescriptorsFromTableStart=offset;offset+=ranges[j].NumDescriptors;
            }
            D3D12_ROOT_PARAMETER parameters[2]{};parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_CBV;parameters[0].Descriptor={description.constantBufferRegisterIndex,description.constantBufferAndSamplersSpaceIndex};
            parameters[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;parameters[1].DescriptorTable={pipeline.resourceRangesNum,ranges};
            std::vector<D3D12_STATIC_SAMPLER_DESC> samplers(description.samplersNum);
            for(UINT j=0;j<samplers.size();++j){auto& s=samplers[j];s.Filter=description.samplers[j]==nrd::Sampler::NEAREST_CLAMP?D3D12_FILTER_MIN_MAG_MIP_POINT:D3D12_FILTER_MIN_MAG_MIP_LINEAR;
                s.AddressU=s.AddressV=s.AddressW=D3D12_TEXTURE_ADDRESS_MODE_CLAMP;s.MaxLOD=D3D12_FLOAT32_MAX;s.ComparisonFunc=D3D12_COMPARISON_FUNC_ALWAYS;s.ShaderRegister=description.samplersBaseRegisterIndex+j;s.RegisterSpace=description.constantBufferAndSamplersSpaceIndex;}
            D3D12_ROOT_SIGNATURE_DESC root{};root.NumParameters=2;root.pParameters=parameters;root.NumStaticSamplers=UINT(samplers.size());root.pStaticSamplers=samplers.data();
            ComPtr<ID3DBlob> code,error;check_hr(D3D12SerializeRootSignature(&root,D3D_ROOT_SIGNATURE_VERSION_1,&code,&error),"Serialize NRD root");
            check_hr(context->device()->CreateRootSignature(0,code->GetBufferPointer(),code->GetBufferSize(),IID_PPV_ARGS(&pipelines[i].root)),"Create NRD root");
            D3D12_COMPUTE_PIPELINE_STATE_DESC state{};state.pRootSignature=pipelines[i].root.Get();state.CS={pipeline.computeShaderDXIL.bytecode,SIZE_T(pipeline.computeShaderDXIL.size)};
            pipelines[i].state=context->compute_pipeline((L"RELAX-"+std::to_wstring(i)).c_str(),state);
        }
        construction_guard.release();
    }
};
DxrNrd::DxrNrd(std::shared_ptr<D3d12Context> context):impl_(std::make_unique<Impl>(std::move(context))){}
DxrNrd::~DxrNrd()=default;
void DxrNrd::resize(UINT width,UINT height,bool validation) {
    auto& i=*impl_;
    if(validation && (!i.validation || i.width!=width || i.height!=height))i.validation=i.context->texture(width,height,DXGI_FORMAT_R8G8B8A8_UNORM);
    if(!validation)i.validation.reset();
    if(i.width==width && i.height==height)return;i.width=width;i.height=height;
    const auto& desc=*nrd::GetInstanceDesc(*i.instance);
    auto create=[&](std::vector<D3d12ResourcePtr>& textures,const nrd::TextureDesc* descriptions,UINT count) {
        textures.clear();for(UINT j=0;j<count;++j){const auto& t=descriptions[j];const UINT scale=std::max(1u,UINT(t.downsampleFactor));textures.push_back(i.context->texture((width+scale-1)/scale,(height+scale-1)/scale,format(t.format)));}
    };
    create(i.permanent,desc.permanentPool,desc.permanentPoolSize);create(i.transient,desc.transientPool,desc.transientPoolSize);
    i.diffuse=i.context->texture(width,height,DXGI_FORMAT_R16G16B16A16_FLOAT);i.specular=i.context->texture(width,height,DXGI_FORMAT_R16G16B16A16_FLOAT);
}
void DxrNrd::dispatch(const DxrFrameConstants& g,const DxrNrdInputs& input,float delta_seconds) {
    auto& i=*impl_;auto& context=*i.context;nrd::CommonSettings common;
    camera_matrices(g.eye,g.forward,g.right,g.up,common.worldToViewMatrix,common.viewToClipMatrix);
    camera_matrices(g.previous_eye,g.previous_forward,g.previous_right,g.previous_up,common.worldToViewMatrixPrev,common.viewToClipMatrixPrev);
    common.motionVectorScale[0]=1.f/i.width;common.motionVectorScale[1]=1.f/i.height;common.motionVectorScale[2]=1;
    common.cameraJitter[0]=g.jitter.x;common.cameraJitter[1]=g.jitter.y;
    common.cameraJitterPrev[0]=i.previous_jitter[0];common.cameraJitterPrev[1]=i.previous_jitter[1];i.previous_jitter[0]=g.jitter.x;i.previous_jitter[1]=g.jitter.y;
    common.resourceSize[0]=common.resourceSizePrev[0]=common.rectSize[0]=common.rectSizePrev[0]=uint16_t(i.width);
    common.resourceSize[1]=common.resourceSizePrev[1]=common.rectSize[1]=common.rectSizePrev[1]=uint16_t(i.height);
    common.frameIndex=g.frame.x;common.timeDeltaBetweenFrames=std::max(delta_seconds*1000.f,1.f);
    common.accumulationMode=g.frame.w?nrd::AccumulationMode::CLEAR_AND_RESTART:nrd::AccumulationMode::CONTINUE;
    common.enableValidation=g.options.z==UINT(DxrDebugView::NrdValidation);
    check_nrd(nrd::SetCommonSettings(*i.instance,common),"Set NRD camera");
    nrd::RelaxSettings settings;settings.diffuseMaxAccumulatedFrameNum=g.sampling.z;settings.specularMaxAccumulatedFrameNum=g.sampling.z;
    settings.diffuseMaxFastAccumulatedFrameNum=std::min(6u,g.sampling.z);settings.specularMaxFastAccumulatedFrameNum=std::min(6u,g.sampling.z);
    // ReSTIR already reduces the signal variance. Keep the mandatory prepass
    // for missing lobe distances, with a smaller footprint to retain detail.
    if(g.options.x && g.options.y){settings.diffusePrepassBlurRadius=8.f;settings.specularPrepassBlurRadius=16.f;}
    settings.hitDistanceReconstructionMode=nrd::HitDistanceReconstructionMode::AREA_3X3;
    check_nrd(nrd::SetDenoiserSettings(*i.instance,0,&settings),"Set RELAX settings");
    const nrd::DispatchDesc* dispatches=nullptr;UINT count=0;const nrd::Identifier id=0;
    check_nrd(nrd::GetComputeDispatches(*i.instance,&id,1,dispatches,count),"Get RELAX dispatches");
    auto select=[&](const nrd::ResourceDesc& resource)->D3d12ResourcePtr {
        switch(resource.type) {
            case nrd::ResourceType::IN_MV:return input.motion;
            case nrd::ResourceType::IN_NORMAL_ROUGHNESS:return input.normal_roughness;
            case nrd::ResourceType::IN_VIEWZ:return input.view_z;
            case nrd::ResourceType::IN_DIFF_RADIANCE_HITDIST:return input.diffuse;
            case nrd::ResourceType::IN_SPEC_RADIANCE_HITDIST:return input.specular;
            case nrd::ResourceType::OUT_DIFF_RADIANCE_HITDIST:return i.diffuse;
            case nrd::ResourceType::OUT_SPEC_RADIANCE_HITDIST:return i.specular;
            case nrd::ResourceType::OUT_VALIDATION:return i.validation;
            case nrd::ResourceType::TRANSIENT_POOL:return i.transient.at(resource.indexInPool);
            case nrd::ResourceType::PERMANENT_POOL:return i.permanent.at(resource.indexInPool);
            default:throw std::runtime_error(std::string("Unbound NRD resource: ")+nrd::GetResourceTypeString(resource.type));
        }
    };
    auto* command=context.commands();const auto& table=i.tables[context.frame_index()];
    for(UINT d=0;d<count;++d) {
        const auto& dispatch=dispatches[d];if((d+1)*i.table_stride>table->count)throw std::runtime_error("NRD descriptor table capacity exceeded");
        const UINT start=table->index+d*i.table_stride;
        for(UINT r=0;r<dispatch.resourcesNum;++r) {
            const auto& binding=dispatch.resources[r];auto resource=select(binding);const bool storage=binding.descriptorType==nrd::DescriptorType::STORAGE_TEXTURE;
            if(storage && resource->state==D3D12_RESOURCE_STATE_UNORDERED_ACCESS)context.uav_barrier(resource);
            context.transition(resource,storage?D3D12_RESOURCE_STATE_UNORDERED_ACCESS:D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
            const auto destination=context.descriptors()->cpu(start+r);const auto desc=resource->resource->GetDesc();
            if(storage){D3D12_UNORDERED_ACCESS_VIEW_DESC view{};view.Format=desc.Format;view.ViewDimension=D3D12_UAV_DIMENSION_TEXTURE2D;context.device()->CreateUnorderedAccessView(resource->resource.Get(),nullptr,&view,destination);}
            else {D3D12_SHADER_RESOURCE_VIEW_DESC view{};view.Format=desc.Format;view.ViewDimension=D3D12_SRV_DIMENSION_TEXTURE2D;view.Shader4ComponentMapping=D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;view.Texture2D.MipLevels=1;context.device()->CreateShaderResourceView(resource->resource.Get(),&view,destination);}
        }
        auto& pipeline=i.pipelines.at(dispatch.pipelineIndex);command->SetComputeRootSignature(pipeline.root.Get());command->SetPipelineState(pipeline.state.Get());
        if(dispatch.constantBufferDataSize){auto constants=context.upload_frame(std::span(reinterpret_cast<const std::byte*>(dispatch.constantBufferData),dispatch.constantBufferDataSize));command->SetComputeRootConstantBufferView(0,constants.address());}
        command->SetComputeRootDescriptorTable(1,context.descriptors()->gpu(start));command->Dispatch(dispatch.gridWidth,dispatch.gridHeight,1);
    }
    context.transition(i.diffuse,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);context.transition(i.specular,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
    if(i.validation)context.transition(i.validation,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE);
}
const D3d12ResourcePtr& DxrNrd::diffuse() const{return impl_->diffuse;}
const D3d12ResourcePtr& DxrNrd::specular() const{return impl_->specular;}
const D3d12ResourcePtr& DxrNrd::validation() const{return impl_->validation;}
} // namespace renderer
