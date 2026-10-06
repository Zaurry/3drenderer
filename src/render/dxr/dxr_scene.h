#pragma once
#include "platform/d3d12/d3d12_context.h"
#include "render/dxr/dxr_gpu.h"
#include "render/dxr/dxr_omm.h"
#include "render/interactive/interactive_render_session.h"
#include <map>
#include <tuple>
#include <future>

namespace renderer {
class DxrScene {
public:
    explicit DxrScene(std::shared_ptr<D3d12Context> context) : context_(std::move(context)) {}
    SceneChangeSet update(const RenderSceneSnapshot&, SceneChangeSet, DxrStatistics&,bool enable_omm=false);
    void bind(DxrFrameConstants&) const;
    D3D12_GPU_VIRTUAL_ADDRESS acceleration_structure() const;
    void retain() const;
private:
    struct Geometry { UINT vertex_offset=0, slot_offset=0, count=0;bool fully_bound=true; };
    struct Blas { D3d12ResourcePtr data;DxrOmmGpu omm; };
    using BlasKey=std::tuple<UINT,bool,std::string>;
    using InstanceKey = std::pair<std::uint64_t,std::uint32_t>;
    using LightKey = std::tuple<unsigned,std::uint64_t,std::uint32_t,std::uint32_t>;
    std::shared_ptr<D3d12Context> context_;
    std::uint64_t source_id_=0;
    SceneRevisions revisions_;
    bool initialized_=false,lights_changed_=false,advance_previous_transforms_=false;
    bool omm_enabled_=false,bakes_in_flight_=false;
    UINT instance_count_=0, light_count_=0, environment_light_=0;
    std::vector<Geometry> geometries_;
    std::map<BlasKey,Blas> blas_;
    std::map<BlasKey,std::shared_future<DxrOmmBake>> omm_bakes_;
    std::vector<BlasKey> instance_keys_;
    std::vector<UINT> instance_cull_flags_;
    std::map<InstanceKey,Mat4> previous_transforms_;
    std::vector<LightKey> light_keys_;
    std::vector<D3d12ResourcePtr> textures_;
    D3d12ResourcePtr vertices_,slots_,instances_,materials_,lights_,previous_lights_,light_cdf_,emitter_map_,remap_,reverse_remap_;
    D3d12ResourcePtr environment_,environment_alias_,tlas_,tlas_scratch_;
    DxrFloat4 environment_color_,environment_info_;
    void upload_geometry(const RenderSceneSnapshot&);
    void upload_textures(const RenderSceneSnapshot&);
    void upload_materials(const RenderSceneSnapshot&);
    void upload_instances(const RenderSceneSnapshot&);
    void upload_lights(const RenderSceneSnapshot&);
    void build_acceleration(const RenderSceneSnapshot&,bool rebuild,DxrStatistics&);
};
} // namespace renderer
