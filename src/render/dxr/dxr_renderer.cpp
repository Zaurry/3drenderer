#include "render/dxr/dxr_renderer.h"
#include "render/dxr/dxr_ser_tuner.h"
#include "render/dxr/dxr_scene.h"
#include "render/dxr/dxr_nrd.h"
#include "render/dxr/dxr_streamline.h"
#include "render/ggx_energy_compensation.h"
#include "platform/d3d12/d3d12_frame_graph.h"
#include <cmath>
#include <cstring>

namespace renderer {
namespace {
template<class T> std::span<const std::byte> object_bytes(const T& value){return std::as_bytes(std::span(&value,1));}
DxrFloat4 packed_camera(const Vec3& v,float w=0){return {v.x(),v.y(),v.z(),w};}
float halton(unsigned index,unsigned base){float value=0,scale=1;while(index){scale/=float(base);value+=float(index%base)*scale;index/=base;}return value;}
struct RayPipeline {
    ComPtr<ID3D12StateObject> state;
    D3d12ResourcePtr table;
    D3D12_DISPATCH_RAYS_DESC dispatch{};
};
RayPipeline create_ray_pipeline(D3d12Context& context,ID3D12RootSignature* root,const char* shader,bool omm=false) {
    RayPipeline result;auto bytecode=load_dxr_shader(shader);
    D3D12_DXIL_LIBRARY_DESC library{};library.DXILLibrary={bytecode.data(),bytecode.size()};
    D3D12_HIT_GROUP_DESC hit{};hit.HitGroupExport=L"HitGroup";hit.Type=D3D12_HIT_GROUP_TYPE_TRIANGLES;hit.ClosestHitShaderImport=L"ClosestHit";hit.AnyHitShaderImport=L"AnyHit";
    D3D12_RAYTRACING_SHADER_CONFIG shader_config{24,8};D3D12_RAYTRACING_PIPELINE_CONFIG1 pipeline_config{1,omm?D3D12_RAYTRACING_PIPELINE_FLAG_ALLOW_OPACITY_MICROMAPS:D3D12_RAYTRACING_PIPELINE_FLAG_NONE};
    D3D12_STATE_SUBOBJECT objects[]={
        {D3D12_STATE_SUBOBJECT_TYPE_DXIL_LIBRARY,&library},
        {D3D12_STATE_SUBOBJECT_TYPE_HIT_GROUP,&hit},
        {D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_SHADER_CONFIG,&shader_config},
        {D3D12_STATE_SUBOBJECT_TYPE_RAYTRACING_PIPELINE_CONFIG1,&pipeline_config},
        {D3D12_STATE_SUBOBJECT_TYPE_GLOBAL_ROOT_SIGNATURE,&root}};
    D3D12_STATE_OBJECT_DESC desc{D3D12_STATE_OBJECT_TYPE_RAYTRACING_PIPELINE,UINT(std::size(objects)),objects};
    check_hr(context.device()->CreateStateObject(&desc,IID_PPV_ARGS(&result.state)),"Create DXR pipeline");
    ComPtr<ID3D12StateObjectProperties> properties;check_hr(result.state.As(&properties),"Query DXR shader identifiers");
    std::array<std::byte,192> table{};const wchar_t* names[]={L"RayGeneration",L"Miss",L"HitGroup"};
    for(unsigned i=0;i<3;++i){const void* identifier=properties->GetShaderIdentifier(names[i]);if(!identifier)throw std::runtime_error("DXR shader export is missing");std::memcpy(table.data()+i*64,identifier,D3D12_SHADER_IDENTIFIER_SIZE_IN_BYTES);}
    result.table=context.upload(table);const auto address=result.table->resource->GetGPUVirtualAddress();
    result.dispatch.RayGenerationShaderRecord={address,32};result.dispatch.MissShaderTable={address+64,32,32};result.dispatch.HitGroupTable={address+128,32,32};result.dispatch.Depth=1;return result;
}
ComPtr<ID3D12RootSignature> create_root(ID3D12Device* device) {
    D3D12_ROOT_PARAMETER1 parameters[2]{};parameters[0].ParameterType=D3D12_ROOT_PARAMETER_TYPE_CBV;parameters[0].Descriptor.ShaderRegister=0;
    parameters[1].ParameterType=D3D12_ROOT_PARAMETER_TYPE_SRV;parameters[1].Descriptor.ShaderRegister=0;
    D3D12_VERSIONED_ROOT_SIGNATURE_DESC desc{};desc.Version=D3D_ROOT_SIGNATURE_VERSION_1_1;desc.Desc_1_1.NumParameters=2;desc.Desc_1_1.pParameters=parameters;
    desc.Desc_1_1.Flags=D3D12_ROOT_SIGNATURE_FLAG_CBV_SRV_UAV_HEAP_DIRECTLY_INDEXED|D3D12_ROOT_SIGNATURE_FLAG_SAMPLER_HEAP_DIRECTLY_INDEXED;
    ComPtr<ID3DBlob> blob,error;check_hr(D3D12SerializeVersionedRootSignature(&desc,&blob,&error),"Serialize DXR root signature");
    ComPtr<ID3D12RootSignature> root;check_hr(device->CreateRootSignature(0,blob->GetBufferPointer(),blob->GetBufferSize(),IID_PPV_ARGS(&root)),"Create DXR root signature");return root;
}
}
struct DxrRenderer::Impl {
    std::shared_ptr<D3d12Context> context;
    DxrScene scene;
    ComPtr<ID3D12RootSignature> root;
    RayPipeline ordinary,ser,ordinary_omm,ser_omm;
    std::array<RayPipeline,3> pt,pt_ser,pt_omm,pt_ser_omm;
    ComPtr<ID3D12PipelineState> resolve;
    ComPtr<ID3D12PipelineState> prepare_guides,compose_dlss;
    ComPtr<ID3D12PipelineState> di_initial,di_spatial,di_initial_omm,di_spatial_omm;
    D3d12ResourcePtr di_reservoirs,pt_reservoirs,neighbor_offsets,reuse_debug,bsdf_tables;
    std::unique_ptr<DxrNrd> nrd;
    DxrNrdInputs nrd_inputs;
    DxrDlssInputs dlss_inputs;
    DxrReconstruction active_reconstruction=DxrReconstruction::Reference;
    D3d12ResourcePtr surface[2],color[2],raw,direct,indirect,diffuse,specular;
    DxrFrameConstants previous{};
    DxrRenderSettings settings;
    DxrStatistics statistics;
    DxrSerTuner ser_tuner;
    RenderFrameOutput output;
    UINT width=0,height=0,output_width=0,output_height=0,frame=0,accumulated=0,stationary_frames=0;
    bool reset_pending=true;
    explicit Impl(std::shared_ptr<D3d12Context> c) : context(c?std::move(c):D3d12Context::create()),scene(context) {
        statistics.device=context->capabilities();statistics.enhanced_barriers_active=context->enhanced_barriers_active();root=create_root(context->device());
        ordinary=create_ray_pipeline(*context,root.Get(),"pathtrace.dxil");
        if(statistics.device.ser_supported)ser=create_ray_pipeline(*context,root.Get(),"pathtrace_ser.dxil");
        auto code=load_dxr_shader("resolve.dxil");D3D12_COMPUTE_PIPELINE_STATE_DESC desc{};desc.pRootSignature=root.Get();desc.CS={code.data(),code.size()};
        resolve=context->compute_pipeline(L"resolve",desc);
        code=load_dxr_shader("prepare_guides.dxil");desc.CS={code.data(),code.size()};
        prepare_guides=context->compute_pipeline(L"prepare_guides",desc);
        code=load_dxr_shader("compose_dlss.dxil");desc.CS={code.data(),code.size()};
        compose_dlss=context->compute_pipeline(L"compose_dlss",desc);
        code=load_dxr_shader("di_initial.dxil");desc.CS={code.data(),code.size()};
        di_initial=context->compute_pipeline(L"di_initial",desc);
        code=load_dxr_shader("di_spatial.dxil");desc.CS={code.data(),code.size()};
        di_spatial=context->compute_pipeline(L"di_spatial",desc);
        if(statistics.device.omm_supported) {
            code=load_dxr_shader("di_initial_omm.dxil");desc.CS={code.data(),code.size()};
            di_initial_omm=context->compute_pipeline(L"di_initial_omm",desc);
            code=load_dxr_shader("di_spatial_omm.dxil");desc.CS={code.data(),code.size()};
            di_spatial_omm=context->compute_pipeline(L"di_spatial_omm",desc);
        }
        std::array<float,512> offsets{};
        for(unsigned j=0;j<256;++j){const float radius=std::sqrt((j+.5f)/256),angle=j*2.39996323f;offsets[2*j]=radius*std::cos(angle);offsets[2*j+1]=radius*std::sin(angle);}
        if(!context->recording())context->begin();
        neighbor_offsets=context->upload_buffer(std::as_bytes(std::span(offsets)),2*sizeof(float));
        std::array<float,1056> albedo{};
        std::copy(kGgxDirectionalAlbedoLut.begin(),kGgxDirectionalAlbedoLut.end(),albedo.begin());
        std::copy(kGgxAverageAlbedoLut.begin(),kGgxAverageAlbedoLut.end(),albedo.begin()+1024);
        bsdf_tables=context->upload_buffer(std::as_bytes(std::span(albedo)),sizeof(float));
    }
    ~Impl(){try{context->flush();}catch(...) {}}
    void resize(UINT w,UINT h,UINT ow,UINT oh) {
        if(width==w && height==h && output_width==ow && output_height==oh)return;
        width=w;height=h;output_width=ow;output_height=oh;reset_pending=true;
        for(auto& s:surface)s=context->buffer(UINT64(w)*h*sizeof(DxrGpuSurface),sizeof(DxrGpuSurface),D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        for(auto& c:color)c=context->texture(ow,oh,DXGI_FORMAT_R32G32B32A32_FLOAT);
        raw=context->texture(w,h,DXGI_FORMAT_R32G32B32A32_FLOAT);direct=context->texture(w,h,DXGI_FORMAT_R32G32B32A32_FLOAT);
        indirect=context->texture(w,h,DXGI_FORMAT_R32G32B32A32_FLOAT);diffuse=context->texture(w,h,DXGI_FORMAT_R32G32B32A32_FLOAT);specular=context->texture(w,h,DXGI_FORMAT_R32G32B32A32_FLOAT);
        nrd_inputs.normal_roughness=context->texture(w,h,DXGI_FORMAT_R16G16B16A16_FLOAT);
        nrd_inputs.motion=context->texture(w,h,DXGI_FORMAT_R32G32B32A32_FLOAT);nrd_inputs.view_z=context->texture(w,h,DXGI_FORMAT_R32_FLOAT);
        nrd_inputs.diffuse=context->texture(w,h,DXGI_FORMAT_R16G16B16A16_FLOAT);nrd_inputs.specular=context->texture(w,h,DXGI_FORMAT_R16G16B16A16_FLOAT);
        const UINT reservoir_count=((w+15)/16)*((h+15)/16)*256;
        di_reservoirs=context->buffer(UINT64(reservoir_count)*3*24,24,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        pt_reservoirs=context->buffer(UINT64(reservoir_count)*4*64,64,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS);
        reuse_debug=context->texture(w,h,DXGI_FORMAT_R32G32B32A32_FLOAT);
        dlss_inputs.color=context->texture(w,h,DXGI_FORMAT_R16G16B16A16_FLOAT);
        dlss_inputs.diffuse_albedo=context->texture(w,h,DXGI_FORMAT_R16G16B16A16_FLOAT);
        dlss_inputs.specular_albedo=context->texture(w,h,DXGI_FORMAT_R16G16B16A16_FLOAT);
        dlss_inputs.specular_distance=context->texture(w,h,DXGI_FORMAT_R32_FLOAT);
        dlss_inputs.depth=context->texture(w,h,DXGI_FORMAT_R32_FLOAT);
        dlss_inputs.motion=context->texture(w,h,DXGI_FORMAT_R32G32_FLOAT);
        dlss_inputs.specular_motion=context->texture(w,h,DXGI_FORMAT_R32G32_FLOAT);
        dlss_inputs.normal_roughness=nrd_inputs.normal_roughness;
    }
};
DxrRenderer::DxrRenderer(std::shared_ptr<D3d12Context> context):impl_(std::make_unique<Impl>(std::move(context))){}
DxrRenderer::~DxrRenderer()=default;
void DxrRenderer::reset(const RenderSceneSnapshot&,const RenderSettings&){impl_->reset_pending=true;}
const RenderFrameOutput& DxrRenderer::render(const RenderSceneSnapshot& snapshot,const Camera& camera,const RenderSettings& input,const InteractiveFrameState& state) {
    auto& i=*impl_;auto& context=*i.context;
    const auto settings=sanitize_dxr_settings(input.dxr);
    // The reference path renders at native resolution and accumulates only stationary frames.
    auto* streamline=context.streamline();
    auto reconstruction=streamline?streamline->select(settings.reconstruction):settings.reconstruction==DxrReconstruction::Reference?DxrReconstruction::Reference:DxrReconstruction::NrdTaau;
    if(settings.debug_view==DxrDebugView::NrdValidation)reconstruction=DxrReconstruction::NrdTaau;
    const bool reference=reconstruction==DxrReconstruction::Reference;
    const UINT ow=UINT(std::max(1,input.width)),oh=UINT(std::max(1,input.height));
    UINT w=reference?ow:std::max(1u,UINT(std::round(float(ow)*settings.internal_scale)));
    UINT h=reference?oh:std::max(1u,UINT(std::round(float(oh)*settings.internal_scale)));
    if(reconstruction==DxrReconstruction::DlssRr || reconstruction==DxrReconstruction::NrdDlss){const auto size=streamline->render_size(reconstruction,ow,oh);w=size.first;h=size.second;reconstruction=streamline->select(settings.reconstruction);}
    if(i.active_reconstruction!=reconstruction || i.width!=w || i.height!=h || i.output_width!=ow || i.output_height!=oh) {
        context.flush();if(streamline)streamline->release_resources(context);
        if(reconstruction==DxrReconstruction::DlssRr || reference)i.nrd.reset();
        i.reset_pending=true;i.active_reconstruction=reconstruction;
    }
    context.begin();context.timestamp(0);
    i.resize(w,h,ow,oh);
    bool use_nrd=!reference && reconstruction!=DxrReconstruction::DlssRr;
    if(use_nrd){if(!i.nrd)i.nrd=std::make_unique<DxrNrd>(i.context);i.nrd->resize(w,h,settings.debug_view==DxrDebugView::NrdValidation);}
    const bool previous_omm=i.statistics.omm_active;
    const auto changes=i.scene.update(snapshot,state.scene_changes,i.statistics,settings.opacity_micromaps && i.statistics.device.omm_supported);
    const auto invalidating=SceneChange::Geometry|SceneChange::Materials|SceneChange::MaterialBindings|SceneChange::Textures|SceneChange::Environment;
    if(i.settings!=settings || state.camera_cut || state.reset_requested || (reference?changes!=SceneChange::None:(changes&invalidating)!=SceneChange::None))i.reset_pending=true;
    i.settings=settings;context.timestamp(1);
    const UINT current=i.frame%2,old=1-current;
    D3d12FrameGraph graph(context);
    using Graph=D3d12FrameGraph;
    context.retain(i.neighbor_offsets);
    context.retain(i.bsdf_tables);
    DxrFrameConstants g{};i.scene.bind(g);
    g.bsdf_tables.x=i.bsdf_tables->srv_index();
    g.eye=packed_camera(camera.eye());g.forward=packed_camera(camera.forward());g.right=packed_camera(camera.right(),camera.viewport_width());g.up=packed_camera(camera.up(),camera.viewport_height());
    const bool camera_changed=std::memcmp(&g.eye,&i.previous.eye,4*sizeof(DxrFloat4))!=0;
    if(camera_changed && reference)i.reset_pending=true;
    // With fixed geometry, lighting and camera, the unjittered output footprint
    // is constant. Its temporal variation is sampling noise, including glass
    // and glossy reflections. Ramp in anew after any motion or scene edit.
    i.stationary_frames=i.reset_pending || camera_changed || changes!=SceneChange::None
        ?0:std::min(i.stationary_frames+1,64u);
    g.previous_eye=i.previous.eye;g.previous_forward=i.previous.forward;g.previous_right=i.previous.right;g.previous_up=i.previous.up;
    if(i.reset_pending){i.accumulated=0;++i.statistics.history_resets;g.previous_eye=g.eye;g.previous_forward=g.forward;g.previous_right=g.right;g.previous_up=g.up;}
    if(i.reset_pending || previous_omm!=i.statistics.omm_active)i.ser_tuner.reset();
    const auto completed=context.timings();
    i.ser_tuner.observe(context.completed_timing_tag(),completed[2]+completed[4]+completed[5]+completed[6]);
    g.jitter={halton(i.frame%64+1,2)-.5f,halton(i.frame%64+1,3)-.5f,float(i.accumulated),0};
    g.previous_jitter=i.reset_pending?g.jitter:i.previous.jitter;
    g.size={w,h,ow,oh};g.frame={i.frame,UINT(settings.samples_per_pixel),UINT(settings.max_bounces),i.reset_pending?1u:0u};
    g.guides={i.surface[current]->uav_index(),i.surface[old]->srv_index(),i.raw->uav_index(),i.direct->uav_index()};
    g.signals={i.indirect->uav_index(),i.diffuse->uav_index(),i.specular->uav_index(),i.color[current]->uav_index()};
    const UINT reconstruction_pass=reference?0:reconstruction==DxrReconstruction::DlssRr?3:reconstruction==DxrReconstruction::NrdDlss?2:1;
    g.history.x=i.di_reservoirs->uav_index();g.history.y=i.pt_reservoirs->uav_index();g.history.z=i.color[old]->srv_index();g.options={settings.restir_di?1u:0u,settings.restir_pt?1u:0u,UINT(settings.debug_view),reconstruction_pass};
    g.reservoirs={((w+15)/16)*256,((w+15)/16)*((h+15)/16)*256,i.neighbor_offsets->srv_index(),i.reuse_debug->uav_index()};
    g.sampling={UINT(settings.di_candidates),UINT(settings.spatial_samples),UINT(settings.history_length),0};
    g.path_sampling={UINT(settings.pt_spatial_samples),UINT(settings.pt_disocclusion_samples),i.stationary_frames,settings.full_resolution_materials?1u:0u};
    if(settings.specular_antialiasing)g.sampling.w|=2;
    // Cached path tails are invalid after geometry motion or a light edit.
    // DI has stable light remapping; PT restarts its tail history on these edits.
    if(has_scene_change(changes,SceneChange::InstanceTransforms) || has_scene_change(changes,SceneChange::Lighting))g.sampling.w|=1;
    if(!reference) {
        g.reconstruction={i.nrd_inputs.normal_roughness->uav_index(),i.nrd_inputs.view_z->uav_index(),i.nrd_inputs.motion->uav_index(),use_nrd && i.nrd->validation()?i.nrd->validation()->srv_index():0};
        g.denoised={use_nrd?i.nrd->diffuse()->srv_index():0,use_nrd?i.nrd->specular()->srv_index():0,i.nrd_inputs.diffuse->uav_index(),i.nrd_inputs.specular->uav_index()};
        g.dlss={i.dlss_inputs.diffuse_albedo->uav_index(),i.dlss_inputs.specular_albedo->uav_index(),i.dlss_inputs.specular_distance->uav_index(),i.dlss_inputs.color->uav_index()};
        g.dlss_guides={i.dlss_inputs.motion->uav_index(),i.dlss_inputs.depth->uav_index(),i.dlss_inputs.specular_motion->uav_index(),0};
    }
    auto constants=context.upload_frame(object_bytes(g));auto* command=context.commands();
    command->SetComputeRootSignature(i.root.Get());command->SetComputeRootConstantBufferView(0,constants.address());command->SetComputeRootShaderResourceView(1,i.scene.acceleration_structure());
    const bool ser_candidate=settings.shader_execution_reordering && i.statistics.device.ser_supported && i.statistics.device.ser_reorders;
    const bool use_ser=ser_candidate && i.ser_tuner.use_ser();
    context.set_timing_tag(i.ser_tuner.tag(i.frame,use_ser,ser_candidate && !i.reset_pending && !camera_changed && changes==SceneChange::None && i.statistics.omm_pending==0));
    const bool use_omm=i.statistics.omm_active;
    auto& pipeline=use_omm?(use_ser?i.ser_omm:i.ordinary_omm):(use_ser?i.ser:i.ordinary);
    if(!pipeline.state)pipeline=create_ray_pipeline(context,i.root.Get(),use_ser?"pathtrace_ser.dxil":"pathtrace.dxil",use_omm);
    context.retain(pipeline.table);command->SetPipelineState1(pipeline.state.Get());
    auto dispatch=pipeline.dispatch;dispatch.Width=w;dispatch.Height=h;
    graph.run("Path trace",{{i.surface[current],Graph::UnorderedWrite},{i.raw,Graph::UnorderedWrite},
        {i.direct,Graph::UnorderedWrite},{i.indirect,Graph::UnorderedWrite},{i.diffuse,Graph::UnorderedWrite},{i.specular,Graph::UnorderedWrite}},[&]{command->DispatchRays(&dispatch);});
    context.timestamp(2);
    if(settings.restir_di) {
        graph.run("DI candidates and temporal reuse",{{i.surface[current],Graph::UnorderedRead},{i.surface[old],Graph::ShaderRead},
            {i.di_reservoirs,Graph::UnorderedWrite},{i.reuse_debug,Graph::UnorderedWrite}},[&]{
            command->SetPipelineState(use_omm?i.di_initial_omm.Get():i.di_initial.Get());command->Dispatch((w+7)/8,(h+7)/8,1);});
        graph.run("DI spatial reuse and visibility",{{i.surface[current],Graph::UnorderedRead},{i.di_reservoirs,Graph::UnorderedWrite},
            {i.reuse_debug,Graph::UnorderedWrite},{i.raw,Graph::UnorderedWrite},{i.direct,Graph::UnorderedWrite},
            {i.diffuse,Graph::UnorderedWrite},{i.specular,Graph::UnorderedWrite}},[&]{
            command->SetPipelineState(use_omm?i.di_spatial_omm.Get():i.di_spatial.Get());command->Dispatch((w+7)/8,(h+7)/8,1);});
    }
    context.timestamp(3);
    if(settings.restir_pt) {
        auto& paths=use_omm?(use_ser?i.pt_ser_omm:i.pt_omm):(use_ser?i.pt_ser:i.pt);
        for(unsigned pass=0;pass<3;++pass) {
            if(!paths[pass].state){const std::string shader="pt_"+std::to_string(pass)+(use_ser?"_ser.dxil":".dxil");paths[pass]=create_ray_pipeline(context,i.root.Get(),shader.c_str(),use_omm);}
            auto rays=paths[pass].dispatch;rays.Width=w;rays.Height=h;
            context.retain(paths[pass].table);command->SetPipelineState1(paths[pass].state.Get());
            constexpr const char* names[]={"PT candidates","PT temporal reuse","PT spatial reuse and shading"};
            graph.run(names[pass],{{i.surface[current],Graph::UnorderedRead},{i.surface[old],Graph::ShaderRead},{i.pt_reservoirs,Graph::UnorderedWrite},
                {i.raw,Graph::UnorderedWrite},{i.indirect,Graph::UnorderedWrite},{i.diffuse,Graph::UnorderedWrite},{i.specular,Graph::UnorderedWrite}},[&]{command->DispatchRays(&rays);});
            context.timestamp(4+pass);
        }
    } else {context.timestamp(4);context.timestamp(5);context.timestamp(6);}
    auto restore=[&]{command->SetComputeRootSignature(i.root.Get());command->SetComputeRootConstantBufferView(0,constants.address());command->SetComputeRootShaderResourceView(1,i.scene.acceleration_structure());};
    if(!reference) {
        graph.run("Reconstruction guides",{{i.surface[current],Graph::UnorderedRead},{i.diffuse,Graph::UnorderedRead},{i.specular,Graph::UnorderedRead},
            {i.nrd_inputs.motion,Graph::UnorderedWrite},{i.nrd_inputs.normal_roughness,Graph::UnorderedWrite},{i.nrd_inputs.view_z,Graph::UnorderedWrite},
            {i.nrd_inputs.diffuse,Graph::UnorderedWrite},{i.nrd_inputs.specular,Graph::UnorderedWrite},
            {i.dlss_inputs.diffuse_albedo,Graph::UnorderedWrite},{i.dlss_inputs.specular_albedo,Graph::UnorderedWrite},
            {i.dlss_inputs.specular_distance,Graph::UnorderedWrite},{i.dlss_inputs.motion,Graph::UnorderedWrite},
            {i.dlss_inputs.specular_motion,Graph::UnorderedWrite},{i.dlss_inputs.depth,Graph::UnorderedWrite}},[&]{
            command->SetPipelineState(i.prepare_guides.Get());command->Dispatch((w+7)/8,(h+7)/8,1);});
        context.uav_barrier({});
        if(use_nrd){i.nrd->dispatch(g,i.nrd_inputs,state.delta_seconds);restore();}
    }
    bool used_dlss=false;
    if((reconstruction==DxrReconstruction::DlssRr || reconstruction==DxrReconstruction::NrdDlss) && settings.debug_view==DxrDebugView::Final) {
        DxrDlssInputs inputs=i.dlss_inputs;
        if(reconstruction==DxrReconstruction::DlssRr)inputs.color=i.raw;
        else {context.transition(i.dlss_inputs.color,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);command->SetPipelineState(i.compose_dlss.Get());command->Dispatch((w+7)/8,(h+7)/8,1);context.uav_barrier({});}
        used_dlss=streamline->evaluate(context,reconstruction,g,inputs,i.color[current]);restore();
        if(!used_dlss) {
            // Keep this frame usable, then select the next supported path next frame.
            reconstruction=DxrReconstruction::NrdTaau;g.options.w=1;g.frame.w=1;i.reset_pending=true;
            if(!i.nrd)i.nrd=std::make_unique<DxrNrd>(i.context);i.nrd->resize(w,h);
            g.denoised.x=i.nrd->diffuse()->srv_index();g.denoised.y=i.nrd->specular()->srv_index();
            constants=context.upload_frame(object_bytes(g));
            if(!use_nrd)i.nrd->dispatch(g,i.nrd_inputs,state.delta_seconds);use_nrd=true;restore();
            context.transition(i.raw,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        }
    }
    if(!used_dlss)graph.run("Resolve and TAAU",{{i.surface[current],Graph::UnorderedRead},{i.surface[old],Graph::ShaderRead},
        {i.color[old],Graph::ShaderRead},{i.color[current],Graph::UnorderedWrite},{i.raw,Graph::UnorderedRead},
        {i.direct,Graph::UnorderedRead},{i.indirect,Graph::UnorderedRead},{i.reuse_debug,Graph::UnorderedRead}},[&]{
        command->SetPipelineState(i.resolve.Get());command->Dispatch((ow+7)/8,(oh+7)/8,1);});context.timestamp(7);
    context.transition(i.color[current],D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);context.timestamp(8);context.timestamp(9);
    auto lease=std::make_shared<DxrFrameResource>();lease->context=i.context;lease->color=i.color[current];
    i.output=DxrTextureHandle{lease,int(ow),int(oh),context.next_fence()};
    ++i.frame;++i.accumulated;i.previous=g;i.reset_pending=false;
    auto& stats=i.statistics;stats.active=true;stats.frames=i.frame;stats.internal_width=int(w);stats.internal_height=int(h);stats.ser_active=use_ser;
    stats.ser_probe_complete=ser_candidate && i.ser_tuner.complete();stats.ser_measured_speedup=i.ser_tuner.speedup();
    stats.ser_probe_ms=i.ser_tuner.ser_ms();stats.trace_probe_ms=i.ser_tuner.trace_ms();
    stats.nrd_active=use_nrd;stats.dlss_rr_active=used_dlss && reconstruction==DxrReconstruction::DlssRr;stats.dlss_sr_active=used_dlss && reconstruction==DxrReconstruction::NrdDlss;
    stats.reconstruction=reference?"Reference accumulation":stats.dlss_rr_active?"DLSS Ray Reconstruction (Quality)":stats.dlss_sr_active?"NRD RELAX + DLSS SR (Quality)":"NRD RELAX + TAAU";
    stats.restir_di_active=settings.restir_di;
    stats.restir_pt_active=settings.restir_pt;
    stats.detail=streamline?streamline->reason():"Native DXR NEE/MIS integrator";stats.allocated_bytes=context.allocated_bytes()+(streamline?streamline->allocated_bytes():0);
    if(i.frame%60==1)stats.video_memory_available=context.query_video_memory(stats.video_memory_usage,stats.video_memory_budget);
    stats.pooled_bytes=context.pooled_bytes();stats.resource_creations=context.resource_creations();stats.resource_reuses=context.resource_reuses();stats.pipeline_cache_hits=context.pipeline_cache_hits();
    if(streamline){stats.runtime_modules=streamline->runtime_modules();stats.dlss_runtime_pinned=streamline->runtime_pinned();}
    const auto timings=context.timings();stats.total_ms=timings[0];stats.acceleration_ms=timings[1];stats.gbuffer_ms=timings[2];stats.direct_ms=timings[3];
    stats.pt_initial_ms=timings[4];stats.pt_temporal_ms=timings[5];stats.pt_spatial_ms=timings[6];stats.indirect_ms=timings[4]+timings[5]+timings[6];stats.reconstruction_ms=timings[7];
    stats.presentation_ms=timings[8]+timings[9];return i.output;
}
const RenderFrameOutput& DxrRenderer::output() const{return impl_->output;}
DxrStatistics DxrRenderer::statistics() const{return impl_->statistics;}
void DxrRenderer::readback(Framebuffer& target) {
    auto& i=*impl_;const auto* handle=std::get_if<DxrTextureHandle>(&i.output);if(!handle || !handle->resource)throw std::runtime_error("DXR has no completed image");
    auto& context=*i.context;if(!context.recording())context.begin();
    const auto texture=handle->resource->color;context.transition(texture,D3D12_RESOURCE_STATE_COPY_SOURCE);
    D3D12_PLACED_SUBRESOURCE_FOOTPRINT layout{};UINT64 bytes=0;const auto desc=texture->resource->GetDesc();context.device()->GetCopyableFootprints(&desc,0,1,0,&layout,nullptr,nullptr,&bytes);
    auto readback=context.readback_buffer(bytes);D3D12_TEXTURE_COPY_LOCATION from{};from.pResource=texture->resource.Get();from.Type=D3D12_TEXTURE_COPY_TYPE_SUBRESOURCE_INDEX;
    D3D12_TEXTURE_COPY_LOCATION to{};to.pResource=readback->resource.Get();to.Type=D3D12_TEXTURE_COPY_TYPE_PLACED_FOOTPRINT;to.PlacedFootprint=layout;
    context.commands()->CopyTextureRegion(&to,0,0,0,&from,nullptr);context.transition(texture,D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE|D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE);context.flush();
    std::byte* data=nullptr;D3D12_RANGE range{0,SIZE_T(bytes)};check_hr(readback->resource->Map(0,&range,reinterpret_cast<void**>(&data)),"Map DXR screenshot");
    target.resize(handle->width,handle->height);
    for(int y=0;y<handle->height;++y){auto* row=reinterpret_cast<const DxrFloat4*>(data+layout.Offset+SIZE_T(y)*layout.Footprint.RowPitch);for(int x=0;x<handle->width;++x)target.set_pixel(x,y,Color(row[x].x,row[x].y,row[x].z));}
    const D3D12_RANGE empty{0,0};readback->resource->Unmap(0,&empty);++i.statistics.readbacks;
}
} // namespace renderer
