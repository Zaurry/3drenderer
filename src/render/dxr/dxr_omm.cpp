#include "render/dxr/dxr_omm.h"
#include "render/dxr/dxr_material.h"
#include <algorithm>
#include <cmath>
#include <limits>

namespace renderer {
namespace {
unsigned compact_even(unsigned value) {
    value&=0x55555555;value=(value|(value>>1))&0x33333333;
    value=(value|(value>>2))&0x0f0f0f0f;value=(value|(value>>4))&0x00ff00ff;
    return (value|(value>>8))&0xffff;
}
unsigned prefix_xor(unsigned value){value^=value>>1;value^=value>>2;value^=value>>4;value^=value>>8;return value;}
struct Interval {float minimum=1,maximum=1;};
Interval multiply(Interval a,Interval b) {
    const float values[]={a.minimum*b.minimum,a.minimum*b.maximum,a.maximum*b.minimum,a.maximum*b.maximum};
    return {*std::min_element(std::begin(values),std::end(values)),*std::max_element(std::begin(values),std::end(values))};
}
int wrap_index(int value,int size,TextureWrap wrap) {
    if(wrap==TextureWrap::ClampToEdge)return std::clamp(value,0,size-1);
    const int period=wrap==TextureWrap::MirroredRepeat?2*size:size;
    value=((value%period)+period)%period;return value<size?value:period-1-value;
}
Interval texture_interval(const ImageTexture& texture,std::array<Vec2,3> uv,const TextureTransform& transform,bool alpha) {
    Vec2 lower=Vec2::Constant(std::numeric_limits<float>::infinity()),upper=-lower;
    const float sn=std::sin(transform.rotation),cs=std::cos(transform.rotation);
    for(auto p:uv) {
        p=p.cwiseProduct(transform.scale);p=Vec2(cs*p.x()-sn*p.y(),sn*p.x()+cs*p.y())+transform.offset;
        if(texture.uv_origin()==TextureUvOrigin::BottomLeft)p.y()=1-p.y();
        lower=lower.cwiseMin(p);upper=upper.cwiseMax(p);
    }
    const bool nearest=texture.mag_filter()==TextureFilter::Nearest;
    const int width=texture.width(),height=texture.height();
    if(!lower.allFinite() || !upper.allFinite() || width<=0 || height<=0)return {};
    // Include a numeric guard at texel boundaries. All bilinear combinations
    // inside the microtriangle lie within the extrema of these texels.
    lower=lower.cwiseProduct(Vec2(float(width),float(height)))-Vec2::Constant(nearest?.001f:.501f);
    upper=upper.cwiseProduct(Vec2(float(width),float(height)))+Vec2::Constant(nearest?.001f:.501f);
    if((upper-lower).maxCoeff()>128 || lower.cwiseAbs().maxCoeff()>1e7 || upper.cwiseAbs().maxCoeff()>1e7)return {-1e20f,1e20f};
    const int x0=int(std::floor(lower.x())),x1=int(std::floor(upper.x())),y0=int(std::floor(lower.y())),y1=int(std::floor(upper.y()));
    Interval result{std::numeric_limits<float>::infinity(),-std::numeric_limits<float>::infinity()};
    for(int y=y0;y<=y1;++y)for(int x=x0;x<=x1;++x) {
        const auto index=std::size_t(wrap_index(y,height,texture.wrap_t()))*width+wrap_index(x,width,texture.wrap_s());
        const float value=alpha?(texture.alphas().empty()?1:texture.alphas()[index]):texture.pixels()[index].dot(Color(.2126f,.7152f,.0722f));
        if(!std::isfinite(value))return {-1e20f,1e20f};result.minimum=std::min(result.minimum,value);result.maximum=std::max(result.maximum,value);
    }
    return result;
}
template<class T>void append(std::string& target,const T& value){target.append(reinterpret_cast<const char*>(&value),sizeof(value));}
template<class T>std::span<const std::byte> bytes(const std::vector<T>& data){return std::as_bytes(std::span(data));}
}
std::array<Vec2,3> dxr_microtriangle(unsigned index,unsigned level) {
    if(!level)return {Vec2(0,0),Vec2(1,0),Vec2(0,1)};
    // OC1's triangular space-filling curve, expressed as deinterleaved Gray
    // coordinates. Vertex order is immaterial for the conservative bounds.
    const unsigned b0=compact_even(index),b1=compact_even(index>>1),fx=prefix_xor(b0),fy=prefix_xor(b0&~b1),t=fy^b1,mask=(1u<<level)-1;
    const unsigned u=((fx&~t)|(b0&~t)|(~b0&~fx&t))&mask,v=(fy^b0)&mask,w=((~fx&~t)|(b0&~t)|(~b0&fx&t))&mask;
    const bool flipped=((u^v^w)&1)==0;const float scale=1.f/float(1u<<level);
    if(flipped)return {Vec2(float(u),float(v+1))*scale,Vec2(float(u+1),float(v+1))*scale,Vec2(float(u+1),float(v))*scale};
    return {Vec2(float(u),float(v))*scale,Vec2(float(u+1),float(v))*scale,Vec2(float(u),float(v+1))*scale};
}
std::string dxr_opacity_signature(const RenderSceneSnapshot& scene,const RenderSceneInstanceSnapshot& instance) {
    if(!std::any_of(instance.materials.begin(),instance.materials.end(),[](const auto& m){return dxr_alpha_mode(m)==AlphaMode::Mask;}))return {};
    std::string key;append(key,scene.revisions.textures);append(key,scene.revisions.material_bindings);
    for(const auto& m:instance.materials) {
        append(key,dxr_alpha_mode(m));append(key,m.two_sided);
        if(dxr_alpha_mode(m)!=AlphaMode::Mask)continue;
        append(key,m.opacity);append(key,m.alpha_cutoff);append(key,m.base_color_texture_id);append(key,m.diffuse_texture_id);append(key,m.opacity_texture_id);
        const auto& tr=m.base_color_texture_transform;
        append(key,tr.texcoord);append(key,tr.rotation);append(key,tr.offset.x());append(key,tr.offset.y());append(key,tr.scale.x());append(key,tr.scale.y());
    }
    return key;
}
DxrOmmBake bake_dxr_opacity(const RenderSceneAssetSnapshot& asset,const RenderSceneInstanceSnapshot& instance,const std::vector<ImageTexture>& textures) {
    DxrOmmBake result;if(!asset.local_scene)return result;
    const auto& triangles=asset.local_scene->triangles;result.indices.resize(triangles.size(),UINT(-4));
    for(unsigned triangle_index=0;triangle_index<triangles.size();++triangle_index) {
        const auto& triangle=triangles[triangle_index];
        if(triangle_index>=asset.triangle_material_slots.size() || !asset.triangle_material_slots[triangle_index].has_value())continue;
        const unsigned slot=asset.triangle_material_slots[triangle_index].value();if(slot>=instance.materials.size())continue;
        const auto& material=instance.materials[slot];
        if(dxr_alpha_mode(material)==AlphaMode::Opaque && material.two_sided){result.indices[triangle_index]=UINT(-2);continue;}
        if(dxr_alpha_mode(material)!=AlphaMode::Mask)continue;
        D3D12_RAYTRACING_OPACITY_MICROMAP_DESC description{};description.ByteOffset=UINT(result.states.size()*4);description.SubdivisionLevel=DxrOmmBake::subdivision;description.Format=D3D12_RAYTRACING_OPACITY_MICROMAP_FORMAT_OC1_4_STATE;
        result.indices[triangle_index]=UINT(result.descriptions.size());result.descriptions.push_back(description);
        const unsigned micro_count=1u<<(2*DxrOmmBake::subdivision),offset=UINT(result.states.size());result.states.resize(offset+micro_count/16);
        const bool uv1=triangle.vertex(0).has_uv1 && triangle.vertex(1).has_uv1 && triangle.vertex(2).has_uv1;
        const bool colors=triangle.vertex(0).has_color && triangle.vertex(1).has_color && triangle.vertex(2).has_color;
        for(unsigned micro=0;micro<micro_count;++micro) {
            const auto bary=dxr_microtriangle(micro,DxrOmmBake::subdivision);std::array<Vec2,3> texcoord0,texcoord1;
            Interval alpha{1,1};if(colors)alpha={1e20f,-1e20f};
            for(unsigned v=0;v<3;++v) {
                const Vec3 weights(1-bary[v].sum(),bary[v].x(),bary[v].y());texcoord0[v]=texcoord1[v]=Vec2::Zero();float value=0;
                for(unsigned j=0;j<3;++j){const auto& vertex=triangle.vertex(j);texcoord0[v]+=weights[j]*vertex.uv;texcoord1[v]+=weights[j]*(uv1?vertex.uv1:vertex.uv);value+=weights[j]*vertex.alpha;}
                if(colors){alpha.minimum=std::min(alpha.minimum,value);alpha.maximum=std::max(alpha.maximum,value);}
            }
            auto sample=[&](int id,const TextureTransform& transform,bool use_alpha) {
                if(id<0 || std::size_t(id)>=textures.size())return Interval{};
                return texture_interval(textures[id],transform.texcoord==1?texcoord1:texcoord0,transform,use_alpha);
            };
            alpha=multiply(alpha,{material.opacity,material.opacity});alpha=multiply(alpha,sample(material.opacity_texture_id,{},false));
            alpha=multiply(alpha,sample(material.base_color_texture_id,material.base_color_texture_transform,true));
            const float minimum=std::clamp(alpha.minimum,0.f,1.f),maximum=std::clamp(alpha.maximum,0.f,1.f);
            unsigned state=2u+unsigned((minimum+maximum)*.5f>=material.alpha_cutoff);
            if(maximum<material.alpha_cutoff-1e-6f)state=0;
            else if(material.two_sided && minimum>=material.alpha_cutoff+1e-6f)state=1;
            result.states[offset+micro/16]|=state<<((micro%16)*2);++result.counts[state];
        }
    }
    return result;
}
DxrOmmGpu build_dxr_opacity(D3d12Context& context,const DxrOmmBake& baked) {
    DxrOmmGpu result;if(baked.descriptions.empty())return result;
    auto states=context.upload_buffer(bytes(baked.states),0),descriptions=context.upload_buffer(bytes(baked.descriptions),0);
    result.indices=context.upload_buffer(bytes(baked.indices),0);
    D3D12_RAYTRACING_OPACITY_MICROMAP_HISTOGRAM_ENTRY histogram{UINT(baked.descriptions.size()),DxrOmmBake::subdivision,D3D12_RAYTRACING_OPACITY_MICROMAP_FORMAT_OC1_4_STATE};
    D3D12_RAYTRACING_OPACITY_MICROMAP_ARRAY_DESC array{};array.NumOmmHistogramEntries=1;array.pOmmHistogram=&histogram;array.InputBuffer=states->resource->GetGPUVirtualAddress();array.PerOmmDescs={descriptions->resource->GetGPUVirtualAddress(),sizeof(D3D12_RAYTRACING_OPACITY_MICROMAP_DESC)};
    D3D12_BUILD_RAYTRACING_ACCELERATION_STRUCTURE_DESC build{};build.Inputs.Type=D3D12_RAYTRACING_ACCELERATION_STRUCTURE_TYPE_OPACITY_MICROMAP_ARRAY;
    build.Inputs.Flags=D3D12_RAYTRACING_ACCELERATION_STRUCTURE_BUILD_FLAG_PREFER_FAST_TRACE;build.Inputs.NumDescs=1;build.Inputs.DescsLayout=D3D12_ELEMENTS_LAYOUT_ARRAY;build.Inputs.pOpacityMicromapArrayDesc=&array;
    D3D12_RAYTRACING_ACCELERATION_STRUCTURE_PREBUILD_INFO info{};context.device()->GetRaytracingAccelerationStructurePrebuildInfo(&build.Inputs,&info);
    auto scratch=context.buffer(info.ScratchDataSizeInBytes,0,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_UNORDERED_ACCESS);
    result.array=context.buffer(info.ResultDataMaxSizeInBytes,0,D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,D3D12_RESOURCE_STATE_RAYTRACING_ACCELERATION_STRUCTURE);
    build.DestAccelerationStructureData=result.array->resource->GetGPUVirtualAddress();build.ScratchAccelerationStructureData=scratch->resource->GetGPUVirtualAddress();
    context.commands()->BuildRaytracingAccelerationStructure(&build,0,nullptr);context.uav_barrier(result.array);return result;
}
}
