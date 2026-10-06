#include "render/dxr/dxr_scene.h"
#include "render/dxr/dxr_material.h"
#include "render/dxr/dxr_alias_table.h"
#include <bit>
#include <cstring>
#include <numbers>
#include <stdexcept>
#include <set>

namespace renderer {
namespace {
DxrFloat4 packed(const Vec3& v,float w=0) { return {v.x(),v.y(),v.z(),w}; }
template<class T> std::span<const std::byte> bytes(const std::vector<T>& v) {return std::as_bytes(std::span(v));}
void matrix(DxrFloat4 (&out)[3],const Mat4& m) {
    for(int r=0;r<3;++r)out[r]={m(r,0),m(r,1),m(r,2),m(r,3)};
}
Vec3 point(const Mat4& m,const Vec3& p){return (m*Vec4(p.x(),p.y(),p.z(),1)).head<3>();}
unsigned wrap(TextureWrap value){return value==TextureWrap::ClampToEdge?2:value==TextureWrap::MirroredRepeat?1:0;}
float luminance(const Color& c){return std::max(0.f,c.dot(Color(.2126f,.7152f,.0722f)));}
bool opaque(const RenderSceneInstanceSnapshot& instance) {
    if(instance.materials.empty())return false;
    const bool two_sided=instance.materials.front().two_sided;
    return std::all_of(instance.materials.begin(),instance.materials.end(),[&](const Material& m){return dxr_alpha_mode(m)==AlphaMode::Opaque && m.two_sided==two_sided;});
}
}

void DxrScene::upload_geometry(const RenderSceneSnapshot& s) {
    std::vector<DxrGpuVertex> vertices;
    std::vector<UINT> slots;
    geometries_.clear();blas_.clear();omm_bakes_.clear();instance_keys_.clear();
    for(const auto& asset:s.assets) {
        Geometry geometry{static_cast<UINT>(vertices.size()),static_cast<UINT>(slots.size()),0};
        if(asset.local_scene) {
            if(!asset.local_scene->spheres.empty())throw std::runtime_error("DXR snapshot must use canonical instanced sphere geometry");
            geometry.count=static_cast<UINT>(asset.local_scene->triangles.size());
            for(UINT i=0;i<geometry.count;++i) {
                const auto& triangle=asset.local_scene->triangles[i];
                const Vec3 normal=triangle.geometric_normal();
                const bool has_normals=triangle.vertex(0).has_normal && triangle.vertex(1).has_normal && triangle.vertex(2).has_normal;
                const bool has_uv1=triangle.vertex(0).has_uv1 && triangle.vertex(1).has_uv1 && triangle.vertex(2).has_uv1;
                const bool has_colors=triangle.vertex(0).has_color && triangle.vertex(1).has_color && triangle.vertex(2).has_color;
                for(int j=0;j<3;++j) {
                    const auto& v=triangle.vertex(j);const Vec2 uv1=has_uv1?v.uv1:v.uv;
                    DxrGpuVertex p{};p.position=packed(v.position,1);p.normal=packed(has_normals?v.normal:normal);
                    p.uv={v.uv.x(),v.uv.y(),uv1.x(),uv1.y()};
                    if(v.has_tangent)p.tangent={v.tangent.x(),v.tangent.y(),v.tangent.z(),v.tangent.w()};
                    p.color=packed(has_colors?v.color:Color::Ones(),has_colors?v.alpha:1);vertices.push_back(p);
                }
                slots.push_back(i<asset.triangle_material_slots.size() && asset.triangle_material_slots[i].has_value()
                    ?asset.triangle_material_slots[i].value():UINT_MAX);
                geometry.fully_bound&=slots.back()!=UINT_MAX;
            }
        }
        geometries_.push_back(geometry);
    }
    vertices_=context_->upload_buffer(bytes(vertices),sizeof(DxrGpuVertex));
    slots_=context_->upload_buffer(bytes(slots),sizeof(UINT));
}
void DxrScene::upload_textures(const RenderSceneSnapshot& s) {
    textures_.clear();textures_.reserve(s.textures.size());
    for(const auto& texture:s.textures) {
        if(texture.width()<=0 || texture.height()<=0){textures_.push_back({});continue;}
        std::vector<DxrFloat4> pixels(texture.pixels().size());
        for(std::size_t i=0;i<pixels.size();++i)pixels[i]=packed(texture.pixels()[i],texture.alphas().empty()?1:texture.alphas()[i]);
        textures_.push_back(context_->upload_texture(texture.width(),texture.height(),DXGI_FORMAT_R32G32B32A32_FLOAT,16,bytes(pixels)));
    }
}
void DxrScene::upload_materials(const RenderSceneSnapshot& s) {
    std::vector<DxrGpuMaterial> materials;
    auto pack=[&](const Material& m) {
        DxrGpuMaterial p{};p.base_opacity=packed(m.base_color,m.opacity);p.emission_roughness=packed(m.emission,m.roughness);
        p.specular_metallic=packed(m.specular_color,m.metallic);p.optics={m.ior,m.alpha_cutoff,m.bump_scale,m.normal_scale};
        p.pbr={m.specular_factor,m.glossiness,m.occlusion_strength,float(m.type)};
        p.flags={UINT(dxr_alpha_mode(m)),m.two_sided?1u:0u,UINT(m.pbr_workflow),0};
        const int ids[]={m.diffuse_texture_id,m.opacity_texture_id,m.bump_texture_id,m.base_color_texture_id,
            m.metallic_roughness_texture_id,m.normal_texture_id,m.occlusion_texture_id,m.emissive_texture_id,
            m.specular_texture_id,m.specular_color_texture_id,m.specular_glossiness_texture_id};
        const TextureTransform empty;
        const TextureTransform* transforms[]={&empty,&empty,&empty,&m.base_color_texture_transform,&m.metallic_roughness_texture_transform,
            &m.normal_texture_transform,&m.occlusion_texture_transform,&m.emissive_texture_transform,&m.specular_texture_transform,
            &m.specular_color_texture_transform,&m.specular_glossiness_texture_transform};
        for(int k=0;k<DxrTextureCount;++k)if(ids[k]>=0 && std::size_t(ids[k])<textures_.size() && textures_[ids[k]]) {
            const auto& t=s.textures[ids[k]];const auto& tr=*transforms[k];auto& dst=p.textures[k];
            dst.texture=textures_[ids[k]]->srv_index();dst.sampler=context_->sampler(t.mag_filter()==TextureFilter::Nearest,wrap(t.wrap_s()),wrap(t.wrap_t()));
            dst.texcoord=(tr.texcoord==1?1:0)|(t.uv_origin()==TextureUvOrigin::BottomLeft?2:0);dst.rotation=tr.rotation;
            dst.transform={tr.offset.x(),tr.offset.y(),tr.scale.x(),tr.scale.y()};
        }
        materials.push_back(p);
    };
    for(const auto& instance:s.instances){for(const auto& m:instance.materials)pack(m);pack(diagnostic_material());}
    if(materials.empty())pack(diagnostic_material());
    materials_=context_->upload_buffer(bytes(materials),sizeof(DxrGpuMaterial));
}
void DxrScene::upload_instances(const RenderSceneSnapshot& s) {
    std::vector<DxrGpuInstance> instances;UINT material_offset=0,emitter_offset=0;
    std::map<InstanceKey,Mat4> current;
    for(const auto& instance:s.instances) {
        if(instance.asset_index<0 || std::size_t(instance.asset_index)>=geometries_.size())throw std::runtime_error("invalid DXR instance asset");
        const auto& geometry=geometries_[instance.asset_index];DxrGpuInstance gpu{};
        const InstanceKey key{instance.object_id,instance.subobject_id};auto previous=previous_transforms_.find(key);
        matrix(gpu.world,instance.object_to_world);matrix(gpu.inverse,instance.world_to_object);
        matrix(gpu.previous,previous==previous_transforms_.end()?instance.object_to_world:previous->second);
        gpu.geometry={geometry.vertex_offset,geometry.slot_offset,material_offset,emitter_offset};
        gpu.identity={UINT(instance.object_id),UINT(instance.object_id>>32),instance.subobject_id,material_offset+UINT(instance.materials.size())};
        material_offset+=UINT(instance.materials.size())+1;emitter_offset+=geometry.count;
        instances.push_back(gpu);current.emplace(key,instance.object_to_world);
    }
    previous_transforms_=std::move(current);
    instances_=context_->upload_buffer(bytes(instances),sizeof(DxrGpuInstance));
}
void DxrScene::upload_lights(const RenderSceneSnapshot& s) {
    std::vector<DxrGpuLight> lights;std::vector<double> powers;std::vector<LightKey> keys;
    std::vector<UINT> emitters;std::map<std::uint64_t,UINT> rectangles;
    const float pi=std::numbers::pi_v<float>;
    auto add=[&](DxrGpuLight p,LightKey key,double power) {
        lights.push_back(p);keys.push_back(key);powers.push_back(std::max(power,1e-12));
    };
    for(const auto& l:s.point_lights) {
        DxrGpuLight p{};p.position_type=packed(l.position,0);p.radiance_range=packed(l.intensity,l.range);
        p.direction_radius.w=l.source_radius;p.identity.z=l.casts_shadows?2:0;
        add(p,{0,l.stable_id,0,0},4*pi*luminance(l.intensity));
    }
    for(const auto& l:s.spot_lights) {
        DxrGpuLight p{};p.position_type=packed(l.position,1);p.direction_radius=packed(l.direction,l.source_radius);
        p.radiance_range=packed(l.intensity,l.range);p.axis_u_inner.w=std::cos(l.inner_cone_radians);p.axis_v_outer.w=std::cos(l.outer_cone_radians);
        p.identity.z=l.casts_shadows?2:0;add(p,{1,l.stable_id,0,0},2*pi*(1-std::cos(l.outer_cone_radians))*luminance(l.intensity));
    }
    for(const auto& l:s.directional_lights) {
        DxrGpuLight p{};p.position_type.w=2;p.direction_radius=packed(l.direction,l.angular_radius_radians);
        p.radiance_range=packed(l.radiance);p.identity.z=l.casts_shadows?2:0;add(p,{2,l.stable_id,0,0},4*pi*luminance(l.radiance));
    }
    for(const auto& l:s.rect_area_lights) {
        DxrGpuLight p{};p.position_type=packed(l.position,3);p.direction_radius=packed(rect_area_light_emission_direction(l),4*l.axis_u.cross(l.axis_v).norm());
        p.radiance_range=packed(l.radiance);p.axis_u_inner=packed(l.axis_u);p.axis_v_outer=packed(l.axis_v);
        p.identity.z=(l.two_sided?1:0)|(l.casts_shadows?2:0);rectangles[l.stable_id]=UINT(lights.size());
        add(p,{3,l.stable_id,0,0},pi*p.direction_radius.w*luminance(l.radiance)*(l.two_sided?2:1));
    }
    for(UINT i=0;i<s.instances.size();++i) {
        const auto& instance=s.instances[i];const auto& asset=s.assets[instance.asset_index];const auto& geometry=geometries_[instance.asset_index];
        for(UINT j=0;j<geometry.count;++j) {
            UINT index=UINT_MAX;
            if(instance.emissive_light_id && rectangles.contains(instance.emissive_light_id))index=rectangles.at(instance.emissive_light_id);
            else if(j<asset.triangle_material_slots.size() && asset.triangle_material_slots[j].has_value()) {
                const auto slot=asset.triangle_material_slots[j].value();
                if(slot>=instance.materials.size())throw std::runtime_error("invalid DXR material slot");
                const auto& m=instance.materials[slot];
                if(m.emission.maxCoeff()>0) {
                    const auto& triangle=asset.local_scene->triangles[j];const Vec3 a=point(instance.object_to_world,triangle.a());
                    const Vec3 e1=point(instance.object_to_world,triangle.b())-a,e2=point(instance.object_to_world,triangle.c())-a;
                    const float area=.5f*e1.cross(e2).norm();DxrGpuLight p{};p.position_type=packed(a,4);
                    p.direction_radius=packed((instance.normal_to_world*triangle.geometric_normal()).normalized(),area);p.radiance_range=packed(m.emission);
                    p.axis_u_inner=packed(e1,std::bit_cast<float>(i));p.axis_v_outer=packed(e2,std::bit_cast<float>(j));
                    p.identity.z=(m.two_sided?1:0)|(instance.emission_casts_shadows?2:0);index=UINT(lights.size());
                    add(p,{4,instance.object_id,instance.subobject_id,j},pi*area*luminance(m.emission)*(m.two_sided?2:1));
                }
            }
            emitters.push_back(index);
        }
    }
    DxrGpuLight env{};env.position_type.w=5;env.identity.z=2;environment_light_=UINT(lights.size());
    double env_power=4*pi*luminance(s.environment)*s.environment_intensity;
    if(s.environment_map){env_power=0;for(const auto& p:s.environment_map->pixels())env_power+=luminance(p.cwiseProduct(s.environment));env_power*=4*pi*s.environment_intensity/s.environment_map->pixels().size();}
    add(env,{5,0,0,0},env_power);
    double total=0;for(double p:powers)total+=p;double sum=0;std::vector<float> cdf;
    for(double p:powers){sum+=p;cdf.push_back(float(sum/total));}cdf.back()=1;
    std::map<LightKey,UINT> indices;for(UINT i=0;i<keys.size();++i)indices.emplace(keys[i],i);
    std::map<LightKey,UINT> old_indices;for(UINT i=0;i<light_keys_.size();++i)old_indices.emplace(light_keys_[i],i);
    std::vector<UINT> remap,reverse;for(const auto& key:light_keys_){auto it=indices.find(key);remap.push_back(it==indices.end()?UINT_MAX:it->second);}
    for(const auto& key:keys){auto it=old_indices.find(key);reverse.push_back(it==old_indices.end()?UINT_MAX:it->second);}
    light_keys_=std::move(keys);light_count_=UINT(lights.size());
    previous_lights_=lights_;
    lights_=context_->upload_buffer(bytes(lights),sizeof(DxrGpuLight));light_cdf_=context_->upload_buffer(bytes(cdf),sizeof(float));
    emitter_map_=context_->upload_buffer(bytes(emitters),sizeof(UINT));remap_=context_->upload_buffer(bytes(remap),sizeof(UINT));
    reverse_remap_=context_->upload_buffer(bytes(reverse),sizeof(UINT));
}
void DxrScene::build_acceleration(const RenderSceneSnapshot& s,bool rebuild,DxrStatistics& stats) {
    auto* device=context_->device();auto* command=context_->commands();
    std::vector<BlasKey> pending;
    for(const auto& key:instance_keys_) {
        if(blas_.contains(key))continue;
        const auto& geometry=geometries_[std::get<0>(key)];if(!geometry.count)continue;
        D3D12_RAYTRACING_GEOMETRY_DESC desc{};desc.Type=D3D12_RAYTRACING_GEOMETRY_TYPE_TRIANGLES;
        desc.Flags=std::get<1>(key)?D3D12_RAYTRACING_GEOMETRY_FLAG_OPAQUE:D3D12_RAYTRACING_GEOMETRY_FLAG_NO_DUPLICATE_ANYHIT_INVOCATION;
        desc.Triangles.VertexBuffer.StartAddress=vertices_->resource->GetGPUVirtualAddress()+UINT64(geometry.vertex_offset)*sizeof(DxrGpuVertex);
        desc.Triangles.VertexBuffer.StrideInBytes=sizeof(DxrGpuVertex);desc.Triangles.VertexCount=geometry.count*3;
        desc.Triangles.VertexFormat=DXGI_FORMAT_R32G32B32_FLOAT;
        DxrOmmGpu omm;const auto triangles=desc.Triangles;D3D12_RAYTRACING_GEOMETRY_OMM_LINKAGE_DESC linkage{};
        if(!std::get<2>(key).empty()) {
            const auto& baked=omm_bakes_.at(key).get();omm=build_dxr_opacity(*context_,baked);
            linkage.OpacityMicromapArray=omm.array->resource->GetGPUVirtualAddress();linkage.OpacityMicromapIndexBuffer={omm.indices->resource->GetGPUVirtualAddress(),sizeof(UINT)};
            linkage.OpacityMicromapIndexFormat=DXGI_FORMAT_R32_UINT;
            desc.Type=D3D12_RAYTRACING_GEOMETRY_TYPE_OMM_TRIANGLES;desc.OmmTriangles={&triangles,&linkage};
            ++stats.omm_builds;for(unsigned state=0;state<4;++state)stats.omm_states[state]+=baked.counts[state];
        }
        D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build{};
        build.Inputs.Type=D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL;
        build.Inputs.Flags=D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE|D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_COMPACTION;
        build.Inputs.DescsLayout=D3D12_ELEMENTS_LAYOUT_ARRAY;build.Inputs.NumDescs=1;build.Inputs.pGeometryDescs=&desc;
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info{};device->GetRaytracingAccelerationStructurePrebuildInfo(&build.Inputs,&info);
        auto scratch=context_->buffer(info.ScratchDataSizeInBytes,0,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        auto data=context_->buffer(info.ResultDataMaxSizeInBytes,0,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
        build.DestAccelerationStructureData=data->resource->GetGPUVirtualAddress();build.ScratchAccelerationStructureData=scratch->resource->GetGPUVirtualAddress();
        command->BuildRaytracingAccelerationStructure(&build,0,nullptr);context_->uav_barrier(data);blas_.emplace(key,Blas{data,omm});pending.push_back(key);++stats.blas_builds;
    }
    if(!pending.empty()) {
        std::vector<D3D12_GPU_VIRTUAL_ADDRESS> addresses;for(const auto& key:pending)addresses.push_back(blas_.at(key).data->resource->GetGPUVirtualAddress());
        const UINT64 size=pending.size()*sizeof(UINT64);auto sizes=context_->buffer(size,0,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
        D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_DESC post{};post.DestBuffer=sizes->resource->GetGPUVirtualAddress();post.InfoType=D3D12_RAYTRACING_ACCELERATION_STRUCTURE_POSTBUILD_INFO_COMPACTED_SIZE;
        command->EmitRaytracingAccelerationStructurePostbuildInfo(&post,UINT(addresses.size()),addresses.data());context_->transition(sizes,D3D12_RESOURCE_STATE_COPY_SOURCE);
        auto readback=context_->readback_buffer(size);command->CopyBufferRegion(readback->resource.Get(),0,sizes->resource.Get(),0,size);context_->flush();
        UINT64* compact_sizes=nullptr;D3D12_RANGE range{0,SIZE_T(size)};check_hr(readback->resource->Map(0,&range,reinterpret_cast<void**>(&compact_sizes)),"Map BLAS compaction sizes");
        command=context_->begin();
        for(std::size_t i=0;i<pending.size();++i) {
            auto& blas=blas_.at(pending[i]);if(compact_sizes[i]==0 || compact_sizes[i]>=blas.data->bytes)continue;
            auto compact=context_->buffer(compact_sizes[i],0,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
            context_->retain(blas.data);command->CopyRaytracingAccelerationStructure(compact->resource->GetGPUVirtualAddress(),blas.data->resource->GetGPUVirtualAddress(),D3D12_RAYTRACING_ACCELERATION_STRUCTURE_COPY_MODE_COMPACT);
            context_->uav_barrier(compact);blas.data=compact;
        }
        const D3D12_RANGE empty{0,0};readback->resource->Unmap(0,&empty);rebuild=true;
    }
    std::vector<D3D12_RAYTRACING_INSTANCE_DESC> instances;
    for(UINT i=0;i<s.instances.size();++i) {
        const auto& instance=s.instances[i];auto it=blas_.find(instance_keys_[i]);if(it==blas_.end())continue;
        D3D12_RAYTRACING_INSTANCE_DESC d{};for(int r=0;r<3;++r)for(int c=0;c<4;++c)d.Transform[r][c]=instance.object_to_world(r,c);
        d.InstanceID=i;d.InstanceMask=255;d.Flags=instance_cull_flags_[i];
        d.AccelerationStructure=it->second.data->resource->GetGPUVirtualAddress();instances.push_back(d);
    }
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build{};
    build.Inputs.Type=D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL;build.Inputs.DescsLayout=D3D12_ELEMENTS_LAYOUT_ARRAY;
    build.Inputs.Flags=D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE|D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_ALLOW_UPDATE;
    build.Inputs.NumDescs=UINT(instances.size());auto descriptions=context_->upload_frame(bytes(instances));build.Inputs.InstanceDescs=descriptions.address();
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info{};device->GetRaytracingAccelerationStructurePrebuildInfo(&build.Inputs,&info);
    if(!tlas_ || tlas_->bytes<info.ResultDataMaxSizeInBytes || instance_count_!=instances.size()) {
        tlas_=context_->buffer(info.ResultDataMaxSizeInBytes,0,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);rebuild=true;
    }
    const auto scratch_size=std::max(info.ScratchDataSizeInBytes,info.UpdateScratchDataSizeInBytes);
    if(!tlas_scratch_ || tlas_scratch_->bytes<scratch_size)tlas_scratch_=context_->buffer(scratch_size,0,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    if(!rebuild){build.Inputs.Flags|=D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PERFORM_UPDATE;build.SourceAccelerationStructureData=tlas_->resource->GetGPUVirtualAddress();++stats.tlas_updates;}else ++stats.tlas_builds;
    build.DestAccelerationStructureData=tlas_->resource->GetGPUVirtualAddress();build.ScratchAccelerationStructureData=tlas_scratch_->resource->GetGPUVirtualAddress();
    command->BuildRaytracingAccelerationStructure(&build,0,nullptr);context_->uav_barrier(tlas_);context_->retain(tlas_scratch_);instance_count_=UINT(instances.size());
}
SceneChangeSet DxrScene::update(const RenderSceneSnapshot& s,SceneChangeSet hint,DxrStatistics& stats,bool enable_omm) {
    const auto changes=scene_changes_for_snapshot(source_id_,revisions_,initialized_,s,hint);
    const auto changed=[&](SceneChange c){return has_scene_change(changes,c);};
    const bool geometry=changed(SceneChange::Geometry);
    if(!initialized_ || s.source_id!=source_id_)previous_transforms_.clear();
    if(geometry)upload_geometry(s);
    else if(changed(SceneChange::MaterialBindings)) {
        std::vector<UINT> slots;
        for(UINT asset_index=0;asset_index<s.assets.size();++asset_index){auto& geometry=geometries_[asset_index];geometry.fully_bound=true;const auto& bindings=s.assets[asset_index].triangle_material_slots;
            for(UINT triangle=0;triangle<geometry.count;++triangle){slots.push_back(triangle<bindings.size() && bindings[triangle].has_value()?bindings[triangle].value():UINT_MAX);geometry.fully_bound&=slots.back()!=UINT_MAX;}}
        slots_=context_->upload_buffer(bytes(slots),sizeof(UINT));
    }
    if(changed(SceneChange::Textures))upload_textures(s);
    if(geometry || changed(SceneChange::Materials) || changed(SceneChange::MaterialBindings) || changed(SceneChange::Textures))upload_materials(s);
    const bool instances_changed=geometry || changed(SceneChange::InstanceTransforms) || changed(SceneChange::MaterialBindings) || changed(SceneChange::Materials);
    if(instances_changed || advance_previous_transforms_)upload_instances(s);
    // Advance once after motion stops, then retain the immutable instance buffer.
    advance_previous_transforms_=changed(SceneChange::InstanceTransforms);
    lights_changed_=geometry || changed(SceneChange::InstanceTransforms) || changed(SceneChange::Lighting) || changed(SceneChange::Materials) || changed(SceneChange::MaterialBindings) || changed(SceneChange::Textures) || changed(SceneChange::Environment);
    if(lights_changed_)upload_lights(s);else previous_lights_=lights_;
    if(changed(SceneChange::Environment) || changed(SceneChange::Textures)) {
        environment_.reset();environment_alias_.reset();
        environment_color_=packed(s.environment*s.environment_intensity,s.environment_intensity);
        environment_info_={s.environment_rotation_degrees*std::numbers::pi_v<float>/180,s.environment_background_visible?1.f:0.f,0,0};
        if(s.environment_map) {
            const auto& e=*s.environment_map;std::vector<DxrFloat4> pixels;for(const auto& p:e.pixels())pixels.push_back(packed(p,1));
            environment_=context_->upload_texture(e.width(),e.height(),DXGI_FORMAT_R32G32B32A32_FLOAT,16,bytes(pixels));
            const auto alias=make_dxr_alias_table(e.importance_pmf());
            environment_alias_=context_->upload_buffer(bytes(alias),sizeof(DxrAliasEntry));environment_info_.z=float(e.width());environment_info_.w=float(e.height());
        }
    }
    // Camera-only frames keep the same GPU scene. Do not rescan every material
    // and allocate signature/key containers once asynchronous baking has settled.
    if(changes==SceneChange::None && enable_omm==omm_enabled_ && stats.omm_pending==0 && !bakes_in_flight_) {
        retain();return changes;
    }
    omm_enabled_=enable_omm;
    std::vector<BlasKey> keys;std::vector<UINT> cull_flags;std::set<BlasKey> wanted_bakes;stats.omm_active=false;stats.omm_pending=0;
    unsigned running_bakes=0;for(const auto& [key,future]:omm_bakes_)if(future.wait_for(std::chrono::seconds(0))!=std::future_status::ready)++running_bakes;
    for(const auto& instance:s.instances) {
        const bool assigned=!instance.materials.empty() && geometries_[instance.asset_index].fully_bound;
        const bool one_sided=assigned && std::all_of(instance.materials.begin(),instance.materials.end(),[](const auto& material){return !material.two_sided;});
        cull_flags.push_back(one_sided?D3D12_RAYTRACING_INSTANCE_FLAG_NONE:D3D12_RAYTRACING_INSTANCE_FLAG_TRIANGLE_CULL_DISABLE);
        BlasKey key{UINT(instance.asset_index),assigned && opaque(instance),{}};
        if(enable_omm) {
            const std::string signature=dxr_opacity_signature(s,instance);
            if(!signature.empty()) {
                const BlasKey candidate{UINT(instance.asset_index),false,signature};auto it=omm_bakes_.find(candidate);
                wanted_bakes.insert(candidate);
                if(it==omm_bakes_.end() && running_bakes<2) {
                    // Baking owns immutable input copies; rendering uses ordinary
                    // AnyHit until the corresponding completed map is ready.
                    auto future=std::async(std::launch::async,[asset=s.assets[instance.asset_index],instance,textures=s.textures]{return bake_dxr_opacity(asset,instance,textures);}).share();
                    it=omm_bakes_.emplace(candidate,std::move(future)).first;
                    ++running_bakes;
                }
                if(it!=omm_bakes_.end() && it->second.wait_for(std::chrono::seconds(0))==std::future_status::ready) {
                    if(!it->second.get().descriptions.empty()){key=candidate;stats.omm_active=true;}
                }else ++stats.omm_pending;
            }
        }
        keys.push_back(std::move(key));
    }
    const bool linkage_changed=keys!=instance_keys_ || cull_flags!=instance_cull_flags_;instance_keys_=std::move(keys);instance_cull_flags_=std::move(cull_flags);
    if(geometry || changed(SceneChange::InstanceTransforms) || linkage_changed)build_acceleration(s,geometry,stats);
    // Fence retention protects submitted frames while stale alpha signatures are
    // evicted. Ordinary BLASes stay cached for instant OMM-off fallback.
    for(auto it=blas_.begin();it!=blas_.end();) {
        if(!std::get<2>(it->first).empty() && std::find(instance_keys_.begin(),instance_keys_.end(),it->first)==instance_keys_.end()) {
            context_->retain(it->second.data);context_->retain(it->second.omm.array);context_->retain(it->second.omm.indices);it=blas_.erase(it);
        }else ++it;
    }
    // Never wait on a stale bake in the render loop. Completed obsolete results
    // can be discarded immediately; at most two immutable jobs remain in flight.
    for(auto it=omm_bakes_.begin();it!=omm_bakes_.end();) {
        if(!wanted_bakes.contains(it->first) && it->second.wait_for(std::chrono::seconds(0))==std::future_status::ready)it=omm_bakes_.erase(it);
        else ++it;
    }
    bakes_in_flight_=running_bakes>0;
    source_id_=s.source_id;revisions_=s.revisions;initialized_=true;retain();return changes;
}
void DxrScene::bind(DxrFrameConstants& g) const {
    g.scene={vertices_->srv_index(),slots_->srv_index(),instances_->srv_index(),materials_->srv_index()};
    g.lighting={lights_->srv_index(),light_count_,environment_?environment_->srv_index():0,environment_alias_?environment_alias_->srv_index():0};
    g.light_extra={emitter_map_->srv_index(),light_cdf_->srv_index(),remap_->srv_index(),environment_light_};g.environment=environment_color_;g.environment_info=environment_info_;
    g.light_history={previous_lights_?previous_lights_->srv_index():lights_->srv_index(),remap_->srv_index(),reverse_remap_->srv_index(),lights_changed_?1u:0u};
}
D3D12_GPU_VIRTUAL_ADDRESS DxrScene::acceleration_structure() const{return tlas_->resource->GetGPUVirtualAddress();}
void DxrScene::retain() const {
    for(const auto& r:{vertices_,slots_,instances_,materials_,lights_,previous_lights_,light_cdf_,emitter_map_,remap_,reverse_remap_,environment_,environment_alias_,tlas_})context_->retain(r);
    for(const auto& t:textures_)context_->retain(t);for(const auto& [key,b]:blas_){context_->retain(b.data);context_->retain(b.omm.array);context_->retain(b.omm.indices);}
}
} // namespace renderer
