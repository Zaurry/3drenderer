#include "render/pathtracer/cuda_scene.cuh"
#include "render/realtime/cuda_realtime_renderer.h"

namespace renderer {
namespace {

// All images are top-left-origin, unexposed, linear HDR. Channels 0/1 are
// direct/indirect diffuse illumination, 2 specular, 3 transmission, 4 emission.
constexpr int kSignals = 4;
struct RtSignals { DVec3 c[5]{}; DVec3 direct{}; };
struct RtFiltered { DVec3 c[kSignals]{}; float variance[kSignals]{}; };
struct RtHistory {
    DVec3 color[kSignals]{};
    DVec2 moments[kSignals]{};
    float length[kSignals]{};
};
struct RtInstance {
    DMatrix3x4 current_to_previous;
    DMatrix3x3 normal_to_previous;
    unsigned long long object_id;
    unsigned long long asset_id;
};
struct RtGuide {
    DVec3 normal{}, geometric{}, albedo{}, previous_normal{};
    DVec2 motion{};
    float depth = 0, previous_depth = 0, roughness = 0, hit_distance = 0;
    unsigned long long object_id = 0, asset_id = 0;
    int material = -1, transparent = 0;
};
struct RtOutputGuide {
    DVec3 normal{};
    float depth = 0;
    unsigned long long object_id = 0;
};
struct RtFrame {
    DScene scene;
    DCamera camera, previous_camera;
    const RtInstance* instances;
    RealtimeRenderSettings settings;
    int width, height, output_width, output_height;
    unsigned long long frame_index, seed;
    DVec2 jitter, previous_jitter;
    int valid_history, shading_changed;
    int* error;
};

__device__ float rt_luma(DVec3 c) { return .2126f*c.x + .7152f*c.y + .0722f*c.z; }
__device__ DVec3 rt_safe(DVec3 c) {
    return finite(c) ? v3(fminf(fmaxf(c.x,0),1e15f),fminf(fmaxf(c.y,0),1e15f),fminf(fmaxf(c.z,0),1e15f)) : v3(0,0,0);
}
__device__ DVec3 rt_divide(DVec3 a, DVec3 b) {
    return v3(a.x/fmaxf(b.x,1e-12f),a.y/fmaxf(b.y,1e-12f),a.z/fmaxf(b.z,1e-12f));
}
__device__ DVec3 rt_albedo(const RtGuide& g) {
    return v3(fmaxf(.04f,g.albedo.x),fmaxf(.04f,g.albedo.y),fmaxf(.04f,g.albedo.z));
}
__device__ DVec3 rt_mix(DVec3 a, DVec3 b, float t) { return add(mul(a,1-t),mul(b,t)); }
__device__ DVec2 rt_project(const DCamera& c, DVec3 p, float& depth) {
    const DVec3 d = sub(p,c.eye);
    depth = dot(d,c.forward);
    const float z = fmaxf(depth,1e-10f);
    return { .5f+dot(d,c.right)/(z*c.viewport_width),
             .5f-dot(d,c.up)/(z*c.viewport_height) };
}
__device__ DRay rt_primary(const RtFrame& f, int i) {
    const float u = (float(i%f.width)+.5f+f.jitter.x)/f.width;
    const float v = (float(i/f.width)+.5f+f.jitter.y)/f.height;
    return {f.camera.eye,normalize(add(add(f.camera.forward,
        mul(f.camera.right,(u-.5f)*f.camera.viewport_width)),
        mul(f.camera.up,(.5f-v)*f.camera.viewport_height)))};
}

__global__ void rt_gbuffer(RtFrame f, RtGuide* guides, DCompactHit* primary_hits) {
    const int i = int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i>=f.width*f.height) return;
    RtGuide g{};
    const DRay ray=rt_primary(f,i);
    DCompactHit compact{}; compact.primitive_kind=-1;
    if(intersect_scene_compact(f.scene,ray,0,1e30f,compact,f.error)) {
        DHit hit{}; reconstruct_hit(f.scene,ray,compact,true,hit);
        DMaterial m{};
        if (hit.material_id>=0 && hit.material_id<f.scene.material_count) m=f.scene.materials[hit.material_id];
        else { m.type=int(MaterialType::Diffuse); m.base_color=v3(1,0,1); m.roughness=1; }
        const DSurface s=evaluate_surface(f.scene,m,hit);
        g.normal=s.shading_normal; g.geometric=hit.geometric_normal;
        g.albedo=s.diffuse_color; g.roughness=s.roughness;
        g.material=hit.material_id;
        g.transparent=(m.type==int(MaterialType::Dielectric) || effective_alpha_mode(m)==int(AlphaMode::Blend));
        DVec3 previous_position=hit.position;
        g.previous_normal=g.normal;
        if(hit.instance_index>=0) {
            const RtInstance instance=f.instances[hit.instance_index];
            g.object_id=instance.object_id; g.asset_id=instance.asset_id;
            previous_position=transform_point(instance.current_to_previous,hit.position);
            g.previous_normal=normalize(transform_direction(instance.normal_to_previous,g.normal));
        }
        const DVec2 uv=rt_project(f.camera,hit.position,g.depth);
        const DVec2 old=rt_project(f.previous_camera,previous_position,g.previous_depth);
        g.motion={old.x-uv.x,old.y-uv.y};
    } else {
        compact.primitive_kind=-1;
        // Sky has rotational flow and no finite depth. Never reconstruct it as a surface.
        float z=0,old_z=0;
        const DVec2 uv=rt_project(f.camera,add(f.camera.eye,ray.direction),z);
        const DVec2 old=rt_project(f.previous_camera,add(f.previous_camera.eye,ray.direction),old_z);
        g.motion={old.x-uv.x,old.y-uv.y}; g.previous_depth=old_z>0?0:-1;
    }
    guides[i]=g; primary_hits[i]=compact;
}

__device__ DVec3 rt_cone(DVec3 direction,float angle,DPcgState& rng) {
    if(angle<=0) return direction;
    const float z=1-random_float(rng)*(1-cosf(angle));
    const float phi=2*kPi*random_float(rng);
    const float r=sqrtf(fmaxf(0,1-z*z));
    return tangent_to_world(v3(r*cosf(phi),r*sinf(phi),z),direction);
}
__device__ bool rt_visible(const RtFrame& f,const DHit& hit,DVec3 direction,float distance,bool casts) {
    if(!f.settings.shadows || !casts) return true;
    const DVec3 origin=offset_origin(hit.position,hit.geometric_normal,direction);
    return !occluded_scene(f.scene,{origin,direction},0,distance<1e29f?fmaxf(0,distance*(1-1e-5f)):distance,f.error);
}
struct RtDirect { DVec3 diffuse{}, specular{}; };
__device__ void rt_add_direct(RtDirect& out,const DSurface& surface,DVec3 outgoing,DVec3 incoming,DVec3 radiance) {
    const DPbrEvaluation e=evaluate_pbr(surface,surface.shading_normal,outgoing,incoming);
    const DVec3 incoming_cos=mul(radiance,fmaxf(0,dot(surface.shading_normal,incoming)));
    out.diffuse=add(out.diffuse,product(e.diffuse,incoming_cos));
    out.specular=add(out.specular,product(e.specular,incoming_cos));
}
__device__ RtDirect rt_direct(const RtFrame& f,const DHit& hit,const DSurface& s,DVec3 outgoing,DPcgState& rng) {
    RtDirect out{};
    const int samples=f.settings.light_samples;
    for(int sample=0;sample<samples;++sample) {
        for(int l=0;l<f.scene.directional_light_count;++l) {
            const DDirectionalLight light=f.scene.directional_lights[l];
            if(!usable(light.direction)) continue;
            const DVec3 dir=rt_cone(normalize(mul(light.direction,-1)),
                f.settings.soft_shadows?light.angular_radius:0,rng);
            if(dot(s.shading_normal,dir)>0 && rt_visible(f,hit,dir,1e30f,light.casts_shadows!=0))
                rt_add_direct(out,s,outgoing,dir,mul(light.radiance,1.0f/samples));
        }
        // Punctual intensity and range retain their existing meaning. Source
        // radius samples visibility over a disk; it does not change light power.
        for(int kind=0;kind<2;++kind) {
            const int count=kind==0?f.scene.point_light_count:f.scene.spot_light_count;
            for(int l=0;l<count;++l) {
                DVec3 position{},intensity{}; float range=0,radius=0; int casts=1;
                if(kind==0) { auto light=f.scene.point_lights[l]; position=light.position; intensity=light.intensity; range=light.range; radius=light.source_radius; casts=light.casts_shadows; }
                else { auto light=f.scene.spot_lights[l]; position=light.position; intensity=light.intensity; range=light.range; radius=light.source_radius; casts=light.casts_shadows; }
                const DVec3 to=sub(position,hit.position); const float d2=length_squared(to);
                if(d2<1e-12f) continue;
                const float d=sqrtf(d2); const DVec3 dir=divv(to,d);
                float factor=punctual_range_attenuation(d,range)/(d2*samples);
                if(kind==1) {
                    const auto light=f.scene.spot_lights[l];
                    const float c=dot(mul(dir,-1),normalize(light.direction));
                    const float cone=saturate((c-light.outer_cosine)/fmaxf(light.inner_cosine-light.outer_cosine,1e-6f));
                    factor*=light.inner_cosine<=light.outer_cosine ? (c>=light.outer_cosine?1.0f:0.0f) : cone;
                }
                if(factor<=0 || dot(s.shading_normal,dir)<=0) continue;
                DVec3 shadow_dir=dir; float shadow_distance=d;
                if(f.settings.soft_shadows && radius>0) {
                    DVec3 t,b; tangent_basis(dir,t,b);
                    const float r=radius*sqrtf(random_float(rng)),phi=2*kPi*random_float(rng);
                    const DVec3 delta=add(to,add(mul(t,r*cosf(phi)),mul(b,r*sinf(phi))));
                    shadow_distance=sqrtf(length_squared(delta)); shadow_dir=divv(delta,shadow_distance);
                }
                if(rt_visible(f,hit,shadow_dir,shadow_distance,casts!=0))
                    rt_add_direct(out,s,outgoing,dir,mul(intensity,factor));
            }
        }
        DShadowTask task{};
        const float ep=environment_strategy_probability(f.scene);
        const bool environment=ep>0 && (ep>=1 || random_float(rng)<ep);
        const bool sampled=environment
            ? sample_environment_shadow_task(f.scene,hit,s,outgoing,v3(1,1,1),0,rng,task)
            : sample_emissive_shadow_task(f.scene,hit,s,outgoing,v3(1,1,1),0,rng,task);
        if(sampled && (!f.settings.shadows || !task.casts_shadows || !occluded_scene(f.scene,task.ray,0,task.t_max,f.error))) {
            const auto e=evaluate_pbr(s,s.shading_normal,outgoing,task.ray.direction);
            const DVec3 fraction=rt_divide(e.diffuse,e.brdf);
            const DVec3 contribution=mul(task.contribution,1.0f/samples);
            const DVec3 diffuse=product(contribution,fraction);
            out.diffuse=add(out.diffuse,diffuse);
            out.specular=add(out.specular,sub(contribution,diffuse));
        }
    }
    return out;
}

__global__ void rt_trace(RtFrame f,const DCompactHit* primary_hits,RtGuide* guides,RtSignals* signals) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i>=f.width*f.height) return;
    RtSignals output{}; float distances=0; int distance_samples=0;
    for(int sample=0;sample<f.settings.samples_per_pixel;++sample) {
        DPcgState rng{};
        pcg_seed(rng,pixel_seed(i%f.width,i/f.width,f.width,f.seed+f.frame_index*0x9e3779b97f4a7c15ULL+sample*0x85ebca6bULL));
        DRay ray=rt_primary(f,i);
        DVec3 throughput=v3(1,1,1),wd{},ws{},wt{};
        float previous_pdf=0; int previous_delta=1;
        for(int bounce=0;bounce<f.settings.max_bounces;++bounce) {
            DCompactHit compact{};
            bool found=false;
            if(bounce==0) { compact=primary_hits[i]; found=compact.primitive_kind>=0; }
            else found=intersect_scene_compact(f.scene,ray,0,1e30f,compact,f.error);
            if(!found) {
                if(bounce>0 || f.scene.environment_background_visible) {
                    float weight=previous_delta?1:power_heuristic(previous_pdf,environment_strategy_probability(f.scene)*environment_pdf(f.scene,ray.direction));
                    const DVec3 incoming=mul(environment_radiance(f.scene,ray.direction),weight);
                    if(bounce==0) output.c[4]=add(output.c[4],incoming);
                    else { output.c[1]=add(output.c[1],product(wd,incoming)); output.c[2]=add(output.c[2],product(ws,incoming)); output.c[3]=add(output.c[3],product(wt,incoming)); }
                }
                break;
            }
            DHit hit{}; reconstruct_hit(f.scene,ray,compact,true,hit);
            if (hit.material_id<0 || hit.material_id>=f.scene.material_count) {
                const DVec3 diagnostic=v3(1,0,1);
                if (bounce==0) output.c[4]=add(output.c[4],diagnostic);
                else { output.c[1]=add(output.c[1],product(wd,diagnostic)); output.c[2]=add(output.c[2],product(ws,diagnostic)); output.c[3]=add(output.c[3],product(wt,diagnostic)); }
                break;
            }
            const DMaterial material=f.scene.materials[hit.material_id];
            const DSurface surface=evaluate_surface(f.scene,material,hit);
            if(bounce==1) { distances+=hit.t; ++distance_samples; }
            const bool passthrough=effective_alpha_mode(material)==int(AlphaMode::Blend) && random_float(rng)>=surface.opacity;
            if(!passthrough && max_component(surface.emission)>0) {
                const float weight=previous_delta?1:power_heuristic(previous_pdf,emissive_light_pdf_for_hit(f.scene,ray.origin,hit));
                const DVec3 incoming=mul(surface.emission,weight);
                if(bounce==0) output.c[4]=add(output.c[4],incoming);
                else { output.c[1]=add(output.c[1],product(wd,incoming)); output.c[2]=add(output.c[2],product(ws,incoming)); output.c[3]=add(output.c[3],product(wt,incoming)); }
            }
            if(!passthrough && material.type==int(MaterialType::Emissive)) break;
            const DVec3 outgoing=mul(ray.direction,-1);
            if(!passthrough && material.type!=int(MaterialType::Dielectric) && (bounce>0 || f.settings.direct_lighting)) {
                const RtDirect d=rt_direct(f,hit,surface,outgoing,rng);
                if(bounce==0) {
                    output.c[0]=add(output.c[0],d.diffuse); output.c[2]=add(output.c[2],d.specular);
                    output.direct=add(output.direct,add(d.diffuse,d.specular));
                } else {
                    const DVec3 incoming=add(d.diffuse,d.specular);
                    output.c[1]=add(output.c[1],product(wd,incoming)); output.c[2]=add(output.c[2],product(ws,incoming)); output.c[3]=add(output.c[3],product(wt,incoming));
                }
            }
            DRay scattered{}; DVec3 attenuation{}; float pdf=0; int delta=1;
            if(passthrough) { scattered={offset_origin(hit.position,hit.geometric_normal,ray.direction),ray.direction}; attenuation=v3(1,1,1); }
            else if(!scatter(ray,hit,material,surface,rng,attenuation,scattered,pdf,delta)) break;
            if(bounce==0) {
                if(passthrough || material.type==int(MaterialType::Dielectric)) {
                    // Delta transmission and reflection keep their own short history.
                    const bool transmitted=passthrough || dot(scattered.direction,hit.geometric_normal)<0;
                    if(transmitted) wt=f.settings.transmission?attenuation:v3(0,0,0);
                    else ws=f.settings.reflections?attenuation:v3(0,0,0);
                } else {
                    const auto e=evaluate_pbr(surface,surface.shading_normal,outgoing,scattered.direction);
                    const float cosine=fmaxf(0,dot(surface.shading_normal,scattered.direction));
                    wd=f.settings.indirect_diffuse?mul(e.diffuse,cosine/fmaxf(pdf,1e-12f)):v3(0,0,0);
                    ws=f.settings.reflections?mul(e.specular,cosine/fmaxf(pdf,1e-12f)):v3(0,0,0);
                }
            } else { wd=product(wd,attenuation); ws=product(ws,attenuation); wt=product(wt,attenuation); }
            throughput=add(add(wd,ws),wt);
            if(!finite(throughput) || max_component(throughput)<=0) break;
            if(bounce+1>=f.settings.roulette_start) {
                const float p=fminf(.95f,fmaxf(.05f,max_component(throughput)));
                if(random_float(rng)>=p) break;
                wd=divv(wd,p); ws=divv(ws,p); wt=divv(wt,p);
            }
            ray=scattered; previous_pdf=pdf; previous_delta=delta || (bounce==0 && !f.settings.direct_lighting);
        }
    }
    for(int c=0;c<5;++c) output.c[c]=rt_safe(mul(output.c[c],1.0f/f.settings.samples_per_pixel));
    output.direct=rt_safe(mul(output.direct,1.0f/f.settings.samples_per_pixel));
    signals[i]=output;
    guides[i].hit_distance=distance_samples>0?distances/distance_samples:0;
}

__device__ bool rt_same_surface(const RtGuide& a,const RtGuide& b,float normal_limit=.8f) {
    if((a.depth>0)!=(b.depth>0)) return false;
    if(a.depth<=0) return true;
    return a.object_id==b.object_id && a.asset_id==b.asset_id && a.material==b.material &&
        dot(a.normal,b.normal)>normal_limit;
}
__device__ float rt_gradient(const RtGuide* g,int i,int w,int h) {
    const int x=i%w,y=i/w; float dx=1e20f,dy=1e20f;
    if(x>0 && rt_same_surface(g[i],g[i-1])) dx=fminf(dx,fabsf(g[i].depth-g[i-1].depth));
    if(x+1<w && rt_same_surface(g[i],g[i+1])) dx=fminf(dx,fabsf(g[i].depth-g[i+1].depth));
    if(y>0 && rt_same_surface(g[i],g[i-w])) dy=fminf(dy,fabsf(g[i].depth-g[i-w].depth));
    if(y+1<h && rt_same_surface(g[i],g[i+w])) dy=fminf(dy,fabsf(g[i].depth-g[i+w].depth));
    return (dx<1e19f?dx:0)+(dy<1e19f?dy:0);
}
__device__ bool rt_neighbor(const RtGuide* g,int p,int q,int w,float gradient,float normal_limit=.8f) {
    if(!rt_same_surface(g[p],g[q],normal_limit)) return false;
    const int distance=abs(p%w-q%w)+abs(p/w-q/w);
    const float tolerance=.02f*fmaxf(g[p].depth,1e-3f)+gradient*float(distance+1);
    return fabsf(g[p].depth-g[q].depth)<=tolerance;
}
__device__ DVec3 rt_demodulate(DVec3 color,const RtGuide& guide,int channel) {
    return channel<2?rt_divide(color,rt_albedo(guide)):color;
}
__device__ DVec3 rt_modulate(DVec3 color,const RtGuide& guide,int channel) {
    return channel<2?product(color,rt_albedo(guide)):color;
}

__global__ void rt_prepare_signal(RtFrame f,const RtGuide* guides,const RtSignals* raw,RtFiltered* prepared) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i>=f.width*f.height) return;
    RtFiltered result{};
    const RtGuide g=guides[i];
    const float gradient=rt_gradient(guides,i,f.width,f.height);
    float sum[3]{},sum2[3]{},count=0;
    if(f.settings.denoise && f.settings.firefly_filter) {
        // Geometry validation and guide fetches are shared by all signals.
        for(int y=-3;y<=3;++y) for(int x=-3;x<=3;++x) {
            const int qx=i%f.width+x,qy=i/f.width+y;
            if(qx<0 || qx>=f.width || qy<0 || qy>=f.height) continue;
            const int q=qy*f.width+qx;
            if(!rt_neighbor(guides,i,q,f.width,gradient)) continue;
            #pragma unroll
            for(int c=0;c<3;++c) {
                const float l=rt_luma(rt_demodulate(raw[q].c[c],guides[q],c));
                sum[c]+=l;sum2[c]+=l*l;
            }
            count+=1;
        }
    }
    #pragma unroll
    for(int c=0;c<kSignals;++c) {
        DVec3 color=rt_demodulate(raw[i].c[c],g,c);
        if(c<3 && count>=8 && !(c==2 && g.roughness<.06f)) {
            const float mean=sum[c]/count;
            const float limit=mean+f.settings.firefly_sigma*sqrtf(fmaxf(0,sum2[c]/count-mean*mean));
            const float l=rt_luma(color);
            if(l>limit && l>1e-8f) color=mul(color,limit/l);
        }
        result.c[c]=color;
    }
    prepared[i]=result;
}

__device__ bool rt_valid_history(const RtFrame& f,const RtGuide& g,const RtGuide& old,float tolerance) {
    if(g.previous_depth<0 || (g.depth>0)!=(old.depth>0)) return false;
    if(g.depth<=0) return true;
    return g.object_id==old.object_id && g.asset_id==old.asset_id && g.material==old.material &&
        fabsf(g.previous_depth-old.depth)<=tolerance &&
        dot(g.previous_normal,old.normal)>=f.settings.normal_threshold &&
        fabsf(g.roughness-old.roughness)<.2f;
}
__device__ DVec2 rt_history_pixel(const RtFrame& f,int i,const RtGuide& g) {
    return {float(i%f.width)+g.motion.x*f.width+f.jitter.x-f.previous_jitter.x,
            float(i/f.width)+g.motion.y*f.height+f.jitter.y-f.previous_jitter.y};
}

__global__ void rt_temporal(RtFrame f,const RtGuide* guides,const RtGuide* previous_guides,
    const RtFiltered* prepared,const RtHistory* previous,RtHistory* history,
    RtFiltered* temporal,RealtimeDiagnosticPixel* diagnostics) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i>=f.width*f.height) return;
    const RtGuide g=guides[i];
    const DVec2 hp=rt_history_pixel(f,i,g);
    const int bx=int(floorf(hp.x)),by=int(floorf(hp.y));
    const float fx=hp.x-bx,fy=hp.y-by;
    int indices[4]; float weights[4]; int taps=0; float sumw=0;
    const float gradient=rt_gradient(guides,i,f.width,f.height);
    const float tolerance=f.settings.depth_threshold*fmaxf(g.previous_depth,1e-3f)+2*gradient;
    if(f.valid_history && f.settings.denoise && f.settings.temporal) {
        for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
            const int qx=bx+x,qy=by+y;
            if(qx<0 || qy<0 || qx>=f.width || qy>=f.height) continue;
            const int q=qy*f.width+qx; const float weight=(x?fx:1-fx)*(y?fy:1-fy);
            if(weight<=0 || !rt_valid_history(f,g,previous_guides[q],tolerance)) continue;
            indices[taps]=q; weights[taps++]=weight; sumw+=weight;
        }
    }
    RtHistory h{}; RtFiltered t{};
    RealtimeDiagnosticPixel diagnostic{};
    diagnostic.depth=g.depth; diagnostic.motion_x=g.motion.x; diagnostic.motion_y=g.motion.y;
    diagnostic.rejected=sumw<1e-5f?1.0f:0.0f;
    float means[kSignals]{},seconds[kSignals]{},count=0;
    for(int y=-2;y<=2;++y) for(int x=-2;x<=2;++x) {
        const int qx=i%f.width+x,qy=i/f.width+y;
        if(qx<0 || qy<0 || qx>=f.width || qy>=f.height) continue;
        const int q=qy*f.width+qx;
        if(!rt_neighbor(guides,i,q,f.width,gradient)) continue;
        #pragma unroll
        for(int c=0;c<kSignals;++c) {
            const float l=rt_luma(prepared[q].c[c]);means[c]+=l;seconds[c]+=l*l;
        }
        count+=1;
    }
    for(int c=0;c<kSignals;++c) {
        const float mean=means[c]/fmaxf(count,1),second=seconds[c]/fmaxf(count,1);
        const float spatial_variance=fmaxf(0,second-mean*mean),sigma=sqrtf(spatial_variance);
        const DVec3 current=prepared[i].c[c]; const float l=rt_luma(current);
        float alpha=1,reactive=0,length=1; DVec3 old_color{}; DVec2 old_moments{};
        if(sumw>1e-5f) {
            float old_length=0,old_distance=0;
            for(int tap=0;tap<taps;++tap) {
                const float w=weights[tap]/sumw; const int q=indices[tap];
                old_color=add(old_color,mul(previous[q].color[c],w));
                old_moments.x+=previous[q].moments[c].x*w; old_moments.y+=previous[q].moments[c].y*w;
                old_length+=previous[q].length[c]*w; old_distance+=previous_guides[q].hit_distance*w;
            }
            const float old_l=rt_luma(old_color);
            reactive=saturate((fabsf(old_l-mean)/(3*sigma+.15f*fabsf(mean)+.02f)-1)*f.settings.reactive_strength);
            if(c>=2 && g.hit_distance>0 && old_distance>0)
                reactive=fmaxf(reactive,saturate(fabsf(g.hit_distance-old_distance)/fmaxf(g.hit_distance,.001f)-.1f));
            if(f.shading_changed) reactive=fmaxf(reactive,.5f);
            if(f.settings.history_clamping && old_l>1e-8f) {
                const float target=fminf(fmaxf(old_l,fmaxf(0,mean-f.settings.history_sigma*sigma)),mean+f.settings.history_sigma*sigma+.001f);
                old_color=mul(old_color,target/old_l);
                // Keep statistical scale consistent with color clipping.
                const float scale=target/old_l; old_moments.x*=scale; old_moments.y*=scale*scale;
            }
            const int limit=c<2?f.settings.diffuse_history:(c==2?f.settings.specular_history:f.settings.transmission_history);
            length=fminf(float(limit),old_length+1);
            if(g.transparent) length=fminf(length,float(f.settings.transmission_history));
            alpha=fmaxf(1/fmaxf(length,1),reactive);
            length=fminf(length,1/fmaxf(alpha,1e-5f));
        }
        h.color[c]=rt_mix(old_color,current,alpha);
        h.moments[c]={old_moments.x*(1-alpha)+l*alpha,old_moments.y*(1-alpha)+l*l*alpha};
        h.length[c]=length;
        t.c[c]=h.color[c];
        const float v=fmaxf(0,h.moments[c].y-h.moments[c].x*h.moments[c].x);
        t.variance[c]=length<4?fmaxf(v,spatial_variance)*4/fmaxf(length,1):v;
        diagnostic.reactive=fmaxf(diagnostic.reactive,reactive);
        diagnostic.variance=fmaxf(diagnostic.variance,t.variance[c]);
    }
    diagnostic.history=h.length[1]; history[i]=h; temporal[i]=t; diagnostics[i]=diagnostic;
}

__global__ void rt_atrous(RtFrame f,const RtGuide* guides,const RtFiltered* input,RtFiltered* output,
    RtHistory* feedback,int iteration) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i>=f.width*f.height) return;
    const int step=1<<iteration; const RtGuide g=guides[i]; RtFiltered result{};
    const float gradient=rt_gradient(guides,i,f.width,f.height);
    const float kernel[5]={1,4,6,4,1};
    for(int c=0;c<kSignals;++c) {
        const int iterations=c<2?f.settings.diffuse_iterations:(c==2?f.settings.specular_iterations:0);
        const bool delta=(c>=2 && (g.transparent || g.roughness<.06f));
        if(!f.settings.denoise || iteration>=iterations || delta || g.depth<=0) {
            result.c[c]=input[i].c[c]; result.variance[c]=input[i].variance[c];
        } else {
            DVec3 sum{}; float weights=0,variance=0;
            const float lp=rt_luma(input[i].c[c]);
            const float sigma=f.settings.luminance_sigma*sqrtf(fmaxf(input[i].variance[c],1e-8f))+.001f;
            for(int y=-2;y<=2;++y) for(int x=-2;x<=2;++x) {
                const int qx=i%f.width+x*step,qy=i/f.width+y*step;
                if(qx<0 || qx>=f.width || qy<0 || qy>=f.height) continue;
                const int q=qy*f.width+qx; const RtGuide neighbor=guides[q];
                if(!rt_same_surface(g,neighbor,0)) continue;
                const float depth_scale=f.settings.depth_sigma*(gradient*float(step*(abs(x)+abs(y)))+.002f*fmaxf(g.depth,.001f))+1e-6f;
                float w=kernel[x+2]*kernel[y+2]*expf(-fabsf(g.depth-neighbor.depth)/depth_scale);
                w*=powf(fmaxf(0,dot(g.normal,neighbor.normal)),f.settings.normal_power);
                w*=expf(-fabsf(lp-rt_luma(input[q].c[c]))/sigma);
                if(c>=2) {
                    w*=expf(-fabsf(g.roughness-neighbor.roughness)*16);
                    if(g.hit_distance>0 && neighbor.hit_distance>0)
                        w*=expf(-fabsf(g.hit_distance-neighbor.hit_distance)/fmaxf(.1f,g.hit_distance*(.1f+g.roughness)));
                }
                sum=add(sum,mul(input[q].c[c],w)); variance+=input[q].variance[c]*w*w; weights+=w;
            }
            result.c[c]=weights>1e-8f?divv(sum,weights):input[i].c[c];
            result.variance[c]=weights>1e-8f?variance/(weights*weights):input[i].variance[c];
        }
        if(iteration==0) feedback[i].color[c]=result.c[c];
    }
    output[i]=result;
}

__device__ DVec3 rt_composite(const RtSignals& raw,const RtFiltered& filtered,const RtGuide& guide) {
    DVec3 color=raw.c[4];
    for(int c=0;c<kSignals;++c) color=add(color,rt_modulate(filtered.c[c],guide,c));
    return rt_safe(color);
}
__device__ int rt_guide_index(RtFrame f,int x,int y) {
    return min(f.height-1,max(0,y))*f.width+min(f.width-1,max(0,x));
}

__global__ void rt_reconstruct(RtFrame f,const RtGuide* guides,const RtSignals* raw,const RtFiltered* filtered,
    const RealtimeDiagnosticPixel* diagnostics,const DVec3* previous_color,const RtOutputGuide* previous_guide,
    DVec3* color,RtOutputGuide* output_guide) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i>=f.output_width*f.output_height) return;
    const float u=(float(i%f.output_width)+.5f)/f.output_width,v=(float(i/f.output_width)+.5f)/f.output_height;
    // Positive jitter offsets primary sample centers. Reconstruct the unjittered output grid.
    const float px=u*f.width-.5f-f.jitter.x,py=v*f.height-.5f-f.jitter.y;
    const int bx=int(floorf(px)),by=int(floorf(py)); const float fx=px-bx,fy=py-by;
    int center=rt_guide_index(f,int(floorf(px+.5f)),int(floorf(py+.5f)));
    // Keep the representative guide discrete. The current RGB reconstruction
    // still mixes subpixel coverage across boundaries; picking the foreground
    // RGB here would expand silhouettes and bias bright emitter edges.
    const RtGuide g=guides[center]; DVec3 current{}; float sumw=0;
    for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
        const int q=rt_guide_index(f,bx+x,by+y);
        const float w=(x?fx:1-fx)*(y?fy:1-fy);
        current=add(current,mul(rt_composite(raw[q],filtered[q],guides[q]),w)); sumw+=w;
    }
    current=sumw>1e-5f?divv(current,sumw):rt_composite(raw[center],filtered[center],g);
    DVec3 mean{},second{}; float count=0;
    for(int y=-1;y<=1;++y) for(int x=-1;x<=1;++x) {
        const int q=rt_guide_index(f,center%f.width+x,center/f.width+y);
        const DVec3 c=rt_composite(raw[q],filtered[q],guides[q]);
        mean=add(mean,c); second=add(second,product(c,c)); count+=1;
    }
    mean=divv(mean,fmaxf(count,1)); second=divv(second,fmaxf(count,1));
    DVec3 history_color{}; float history_weight=0;
    const bool upscale=f.width!=f.output_width || f.height!=f.output_height;
    const bool temporal_enabled=upscale?f.settings.temporal_upscale:f.settings.taa;
    if(f.valid_history && temporal_enabled && g.previous_depth>=0) {
        // Final history is unjittered: do not apply the input jitter correction twice.
        const float ox=(u+g.motion.x)*f.output_width-.5f,oy=(v+g.motion.y)*f.output_height-.5f;
        const int x0=int(floorf(ox)),y0=int(floorf(oy)); const float tx=ox-x0,ty=oy-y0;
        for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
            const int qx=x0+x,qy=y0+y;
            if(qx<0 || qy<0 || qx>=f.output_width || qy>=f.output_height) continue;
            const int q=qy*f.output_width+qx; const RtOutputGuide old=previous_guide[q];
            bool valid=(g.depth>0)==(old.depth>0);
            if(g.depth>0) valid=valid && g.object_id==old.object_id && dot(g.previous_normal,old.normal)>f.settings.normal_threshold &&
                fabsf(g.previous_depth-old.depth)<=f.settings.depth_threshold*fmaxf(g.previous_depth,1e-3f)+2*rt_gradient(guides,center,f.width,f.height);
            if(!valid) continue;
            const float w=(x?tx:1-tx)*(y?ty:1-ty); history_color=add(history_color,mul(previous_color[q],w)); history_weight+=w;
        }
    }
    if(history_weight>1e-5f) {
        history_color=divv(history_color,history_weight);
        const DVec3 variance=sub(second,product(mean,mean));
        const DVec3 sigma=v3(sqrtf(fmaxf(0,variance.x)),sqrtf(fmaxf(0,variance.y)),sqrtf(fmaxf(0,variance.z)));
        const float k=f.settings.taa_clip_sigma;
        history_color=v3(
            fminf(fmaxf(history_color.x,mean.x-k*sigma.x),mean.x+k*sigma.x),
            fminf(fmaxf(history_color.y,mean.y-k*sigma.y),mean.y+k*sigma.y),
            fminf(fmaxf(history_color.z,mean.z-k*sigma.z),mean.z+k*sigma.z));
        float alpha=fmaxf(f.settings.taa_current_weight,diagnostics[center].reactive);
        if(g.transparent) alpha=fmaxf(alpha,.5f);
        current=rt_mix(history_color,current,alpha);
    }
    color[i]=rt_safe(current); output_guide[i]={g.normal,g.depth,g.object_id};
}

__global__ void rt_present(RtFrame f,const DVec3* resolved,const RtGuide* guides,const RtSignals* raw,
    const RtFiltered* temporal,const RtFiltered* filtered,const RealtimeDiagnosticPixel* diagnostics,
    DVec3* output,cudaSurfaceObject_t surface) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i>=f.output_width*f.output_height) return;
    const int x=i%f.output_width,y=i/f.output_width;
    const int p=rt_guide_index(f,int((float(x)+.5f)*f.width/f.output_width),int((float(y)+.5f)*f.height/f.output_height));
    DVec3 c=resolved[i]; const RtGuide g=guides[p]; const auto d=diagnostics[p];
    const auto view=f.settings.debug_view;
    if(view==RealtimeDebugView::Final && f.settings.sharpening>0) {
        DVec3 neighbors{}; float count=0;
        for(int oy=-1;oy<=1;++oy) for(int ox=-1;ox<=1;++ox) {
            const int qx=x+ox,qy=y+oy;
            if(qx<0 || qy<0 || qx>=f.output_width || qy>=f.output_height) continue;
            neighbors=add(neighbors,resolved[qy*f.output_width+qx]); count+=1;
        }
        c=add(c,mul(sub(c,divv(neighbors,count)),f.settings.sharpening));
    }
    if(view==RealtimeDebugView::Raw) { c={}; for(int n=0;n<5;++n)c=add(c,raw[p].c[n]); }
    else if(view==RealtimeDebugView::Direct) c=raw[p].direct;
    else if(view==RealtimeDebugView::IndirectDiffuse) c=rt_modulate(filtered[p].c[1],g,1);
    else if(view==RealtimeDebugView::Reflection) c=filtered[p].c[2];
    else if(view==RealtimeDebugView::Transmission) c=filtered[p].c[3];
    else if(view==RealtimeDebugView::Albedo) c=g.albedo;
    else if(view==RealtimeDebugView::Normal) c=g.depth>0?add(mul(g.normal,.5f),v3(.5f,.5f,.5f)):v3(0,0,0);
    else if(view==RealtimeDebugView::Depth) c=v3(g.depth/(1+g.depth),g.depth/(1+g.depth),g.depth/(1+g.depth));
    else if(view==RealtimeDebugView::Motion) c=v3(.5f+g.motion.x*8,.5f+g.motion.y*8,.5f);
    else if(view==RealtimeDebugView::Variance) { float v=sqrtf(d.variance); c=v3(v/(1+v),v/(1+v),v/(1+v)); }
    else if(view==RealtimeDebugView::HistoryLength) c=v3(d.history/f.settings.diffuse_history,d.history/f.settings.diffuse_history,d.history/f.settings.diffuse_history);
    else if(view==RealtimeDebugView::Rejection) c=v3(d.rejected,0,0);
    else if(view==RealtimeDebugView::Reactive) c=v3(d.reactive,d.reactive,0);
    else if(view==RealtimeDebugView::Temporal) c=rt_composite(raw[p],temporal[p],g);
    else if(view==RealtimeDebugView::Filtered) c=rt_composite(raw[p],filtered[p],g);
    c=rt_safe(c); output[i]=c;
    if(surface) surf2Dwrite(make_float4(c.x,c.y,c.z,1),surface,x*int(sizeof(float4)),y);
}

DCamera rt_camera(const Camera& c) {
    return {to_device(c.eye()),to_device(c.forward()),to_device(c.right()),to_device(c.up()),c.viewport_width(),c.viewport_height()};
}
float rt_halton(unsigned long long index,unsigned base) {
    float f=1,result=0;
    while(index) { f/=float(base); result+=f*float(index%base); index/=base; }
    return result;
}

} // namespace

class CudaRealtimeRenderer::Impl {
public:
    explicit Impl(CudaDeviceContext context):context_(std::move(context)) {
        context_.activate(); statistics_.device_id=context_.device_id();
        check_cuda(cudaStreamCreateWithFlags(&stream_,cudaStreamNonBlocking),"create RTRT stream");
        errors_.resize(1,statistics_);
        check_cuda(cudaMemsetAsync(errors_.get(),0,sizeof(int),stream_),"initialize RTRT errors");
    }
    ~Impl() {
        context_.activate();
        if(stream_) { cudaStreamSynchronize(stream_); scene_.reset(); cudaStreamDestroy(stream_); }
    }
    void reset(const RenderSceneSnapshot& snapshot,const RenderSettings& settings) {
        context_.activate();
        check_cuda(cudaStreamSynchronize(stream_),"reset RTRT stream");
        require_device(settings);
        scene_=std::make_unique<CudaSceneStorage>(snapshot,stream_,statistics_);
        source_=snapshot.source_id; revisions_=snapshot.revisions;
        previous_instances_.clear(); valid_=false;
        statistics_.realtime.active=true;
    }
    void require_device(const RenderSettings& s) const {
        if(s.path.cuda_device!=context_.device_id()) throw std::runtime_error("RTRT CUDA device does not match renderer context");
    }
    void render(const RenderSceneSnapshot& snapshot,const Camera& camera,const RenderSettings& settings,
        const InteractiveFrameState& state,CudaSurfaceHandle surface) {
        context_.activate(); require_device(settings); refresh();
        const auto s=sanitize_realtime_settings(settings.realtime);
        if(settings.width<=0 || settings.height<=0 || std::uint64_t(settings.width)*std::uint64_t(settings.height)>std::uint64_t(INT_MAX))
            throw std::invalid_argument("invalid RTRT output dimensions");
        const int w=std::max(1,int(std::lround(float(settings.width)*s.internal_scale)));
        const int h=std::max(1,int(std::lround(float(settings.height)*s.internal_scale)));
        auto changes=scene_changes_for_snapshot(source_,revisions_,bool(scene_),snapshot,state.scene_changes);
        const bool different_source=source_!=snapshot.source_id;
        if(!scene_) reset(snapshot,settings);
        else if(changes!=SceneChange::None) scene_->sync(snapshot,changes);
        const bool resize=w!=width_ || h!=height_ || settings.width!=output_width_ || settings.height!=output_height_;
        auto history_settings=s; history_settings.debug_view=RealtimeDebugView::Final; history_settings.sharpening=0;
        const bool projection_change=valid_ && (std::abs(previous_camera_.viewport_width-camera.viewport_width())>1e-5f || std::abs(previous_camera_.viewport_height-camera.viewport_height())>1e-5f);
        bool reset_history=!valid_ || different_source || resize || state.camera_cut || state.reset_requested || projection_change ||
            history_settings!=history_settings_ || has_scene_change(changes,SceneChange::Geometry) || has_scene_change(changes,SceneChange::Environment);
        if(resize) {
            check_cuda(cudaStreamSynchronize(stream_),"resize RTRT buffers");
            width_=w; height_=h; output_width_=settings.width; output_height_=settings.height;
            const std::size_t n=std::size_t(w)*h,o=std::size_t(output_width_)*output_height_;
            for(int j=0;j<2;++j) { guides_[j].resize_exact(n,statistics_); history_[j].resize_exact(n,statistics_); filter_[j].resize_exact(n,statistics_); taa_[j].resize_exact(o,statistics_); output_guides_[j].resize_exact(o,statistics_); }
            primary_.resize_exact(n,statistics_); raw_.resize_exact(n,statistics_); prepared_.resize_exact(n,statistics_);
            temporal_.resize_exact(n,statistics_); diagnostics_.resize_exact(n,statistics_); output_.resize_exact(o,statistics_);
            statistics_.realtime.framebuffer_bytes=n*(2*sizeof(RtGuide)+2*sizeof(RtHistory)+4*sizeof(RtFiltered)+sizeof(DCompactHit)+sizeof(RtSignals)+sizeof(RealtimeDiagnosticPixel))+
                o*(3*sizeof(DVec3)+2*sizeof(RtOutputGuide));
        }
        if(reset_history) {
            ++statistics_.realtime.history_resets; valid_=false;
            for(int j=0;j<2;++j) {
                check_cuda(cudaMemsetAsync(history_[j].get(),0,history_[j].size()*sizeof(RtHistory),stream_),"clear RTRT history");
                check_cuda(cudaMemsetAsync(guides_[j].get(),0,guides_[j].size()*sizeof(RtGuide),stream_),"clear RTRT guides");
                check_cuda(cudaMemsetAsync(taa_[j].get(),0,taa_[j].size()*sizeof(DVec3),stream_),"clear TAA history");
                check_cuda(cudaMemsetAsync(output_guides_[j].get(),0,output_guides_[j].size()*sizeof(RtOutputGuide),stream_),"clear TAA guides");
            }
        }
        instance_host_.resize(snapshot.instances.size());
        for(std::size_t j=0;j<snapshot.instances.size();++j) {
            const auto& current=snapshot.instances[j];
            const auto old=previous_instances_.find(current.object_id);
            Mat4 motion=Mat4::Identity(); Mat3 normal=Mat3::Identity();
            if(valid_ && old!=previous_instances_.end()) {
                motion=old->second.first*current.world_to_object;
                normal=old->second.second*current.object_to_world.topLeftCorner<3,3>().transpose();
            }
            instance_host_[j]={to_device_affine(motion),to_device_matrix(normal),current.object_id,
                current.asset_index>=0?snapshot.assets[std::size_t(current.asset_index)].asset_id:0};
        }
        instances_.upload(instance_host_,stream_,statistics_);
        const int write=1-index_;
        const unsigned long long sequence=statistics_.realtime.frames+1;
        const bool jitter_enabled=s.taa || (s.temporal_upscale && s.internal_scale<1);
        const DVec2 jitter=jitter_enabled?DVec2{rt_halton(sequence%1024+1,2)-.5f,rt_halton(sequence%1024+1,3)-.5f}:DVec2{0,0};
        RtFrame f{scene_->view(),rt_camera(camera),valid_?previous_camera_:rt_camera(camera),instances_.get(),s,
            width_,height_,output_width_,output_height_,sequence,settings.path.sample_seed_offset,jitter,previous_jitter_,valid_?1:0,
            (has_scene_change(changes,SceneChange::Lighting)||has_scene_change(changes,SceneChange::Materials)||has_scene_change(changes,SceneChange::MaterialBindings)||has_scene_change(changes,SceneChange::Textures))?1:0,errors_.get()};
        const int blocks=(width_*height_+kThreadsPerBlock-1)/kThreadsPerBlock;
        const int output_blocks=(output_width_*output_height_+kThreadsPerBlock-1)/kThreadsPerBlock;
        // Reuse pending timing events only after they complete; presentation never
        // synchronizes merely to collect statistics.
        const bool timing=!timers_[5].pending();
        if(timing) { timers_[5].begin(stream_); timers_[0].begin(stream_); }
        rt_gbuffer<<<blocks,kThreadsPerBlock,0,stream_>>>(f,guides_[write].get(),primary_.get());
        if(timing) { timers_[0].end(stream_); timers_[1].begin(stream_); }
        rt_trace<<<blocks,kThreadsPerBlock,0,stream_>>>(f,primary_.get(),guides_[write].get(),raw_.get());
        if(timing) { timers_[1].end(stream_); timers_[2].begin(stream_); }
        rt_prepare_signal<<<blocks,kThreadsPerBlock,0,stream_>>>(f,guides_[write].get(),raw_.get(),prepared_.get());
        rt_temporal<<<blocks,kThreadsPerBlock,0,stream_>>>(f,guides_[write].get(),guides_[index_].get(),prepared_.get(),history_[index_].get(),history_[write].get(),temporal_.get(),diagnostics_.get());
        if(timing) { timers_[2].end(stream_); timers_[3].begin(stream_); }
        const RtFiltered* input=temporal_.get();
        const int iterations=s.denoise?std::max(s.diffuse_iterations,s.specular_iterations):0;
        for(int pass=0;pass<iterations;++pass) {
            RtFiltered* out=filter_[pass%2].get();
            rt_atrous<<<blocks,kThreadsPerBlock,0,stream_>>>(f,guides_[write].get(),input,out,history_[write].get(),pass); input=out;
        }
        if(timing) { timers_[3].end(stream_); timers_[4].begin(stream_); }
        rt_reconstruct<<<output_blocks,kThreadsPerBlock,0,stream_>>>(f,guides_[write].get(),raw_.get(),input,diagnostics_.get(),taa_[index_].get(),output_guides_[index_].get(),taa_[write].get(),output_guides_[write].get());
        rt_present<<<output_blocks,kThreadsPerBlock,0,stream_>>>(f,taa_[write].get(),guides_[write].get(),raw_.get(),temporal_.get(),input,diagnostics_.get(),output_.get(),cudaSurfaceObject_t(surface));
        if(timing) { timers_[4].end(stream_); timers_[5].end(stream_); }
        check_cuda(cudaGetLastError(),"RTRT frame kernels");
        // A small asynchronous error mailbox is checked after the recorded event.
        if(!error_pending_) {
            check_cuda(cudaMemcpyAsync(&host_error_,errors_.get(),sizeof(int),cudaMemcpyDeviceToHost,stream_),"RTRT error mailbox");
            if(!error_event_) check_cuda(cudaEventCreateWithFlags(&error_event_,cudaEventDisableTiming),"RTRT error event");
            check_cuda(cudaEventRecord(error_event_,stream_),"record RTRT error event"); error_pending_=true;
        }
        index_=write; valid_=true; previous_camera_=f.camera; previous_jitter_=jitter;
        source_=snapshot.source_id; revisions_=snapshot.revisions; history_settings_=history_settings;
        previous_instances_.clear();
        for(const auto& instance:snapshot.instances) previous_instances_.emplace(instance.object_id,std::make_pair(instance.object_to_world,instance.normal_to_world));
        statistics_.realtime.active=true; ++statistics_.realtime.frames;
        statistics_.internal_width=width_; statistics_.internal_height=height_;
        statistics_.presentation_updated=true; statistics_.work_mode=CudaPathWorkMode::FullFrame;
    }
    void refresh() {
        context_.activate();
        float* values[]={&statistics_.realtime.gbuffer_ms,&statistics_.realtime.lighting_ms,&statistics_.realtime.temporal_ms,
            &statistics_.realtime.filter_ms,&statistics_.realtime.reconstruction_ms,&statistics_.realtime.total_ms};
        for(int j=0;j<6;++j) timers_[j].update(*values[j]);
        statistics_.trace_milliseconds=statistics_.realtime.lighting_ms;
        statistics_.presentation_milliseconds=statistics_.realtime.reconstruction_ms;
        if(scene_) scene_->update_timing();
        if(error_pending_) {
            const auto e=cudaEventQuery(error_event_);
            if(e==cudaSuccess) { error_pending_=false; if(host_error_) throw std::runtime_error("RTRT traversal error: "+std::to_string(host_error_)); }
            else if(e!=cudaErrorNotReady) check_cuda(e,"RTRT error event query");
        }
    }
    void download(Framebuffer& target) {
        if(!valid_) throw std::logic_error("RTRT has no completed frame");
        staging_.resize(output_.size(),statistics_); output_.download(staging_.get(),stream_);
        check_cuda(cudaStreamSynchronize(stream_),"download RTRT output"); refresh();
        target.resize(output_width_,output_height_);
        std::vector<Color> pixels(output_.size());
        for(std::size_t i=0;i<pixels.size();++i) pixels[i]=Color(staging_.get()[i].x,staging_.get()[i].y,staging_.get()[i].z);
        target.set_pixels(std::move(pixels)); ++statistics_.framebuffer_downloads;
    }
    std::vector<RealtimeDiagnosticPixel> diagnostics() {
        std::vector<RealtimeDiagnosticPixel> result(diagnostics_.size()); diagnostics_.download(result.data(),stream_);
        check_cuda(cudaStreamSynchronize(stream_),"download RTRT diagnostics"); refresh(); return result;
    }
    CudaDeviceContext context_;
    CudaPathStatistics statistics_;
    cudaStream_t stream_=nullptr;
    std::unique_ptr<CudaSceneStorage> scene_;
    std::array<DeviceBuffer<RtGuide>,2> guides_;
    std::array<DeviceBuffer<RtHistory>,2> history_;
    std::array<DeviceBuffer<RtFiltered>,2> filter_;
    std::array<DeviceBuffer<DVec3>,2> taa_;
    std::array<DeviceBuffer<RtOutputGuide>,2> output_guides_;
    DeviceBuffer<DCompactHit> primary_;
    DeviceBuffer<RtSignals> raw_;
    DeviceBuffer<RtFiltered> prepared_,temporal_;
    DeviceBuffer<RealtimeDiagnosticPixel> diagnostics_;
    DeviceBuffer<DVec3> output_;
    DeviceBuffer<RtInstance> instances_;
    DeviceBuffer<int> errors_;
    PinnedHostBuffer<DVec3> staging_;
    std::array<CudaEventTimer,6> timers_;
    std::vector<RtInstance> instance_host_;
    std::unordered_map<std::uint64_t,std::pair<Mat4,Mat3>> previous_instances_;
    std::uint64_t source_=0;
    SceneRevisions revisions_{};
    RealtimeRenderSettings history_settings_{};
    DCamera previous_camera_{};
    DVec2 previous_jitter_{};
    int width_=0,height_=0,output_width_=0,output_height_=0,index_=0,host_error_=0;
    bool valid_=false,error_pending_=false;
    struct EventOwner {
        cudaEvent_t event=nullptr;
        ~EventOwner(){if(event)cudaEventDestroy(event);}
    } error_owner_;
    cudaEvent_t& error_event_=error_owner_.event;
};

CudaRealtimeRenderer::CudaRealtimeRenderer(CudaDeviceContext c):impl_(std::make_unique<Impl>(std::move(c))) {}
CudaRealtimeRenderer::~CudaRealtimeRenderer()=default;
void CudaRealtimeRenderer::reset(const RenderSceneSnapshot& s,const RenderSettings& r){impl_->reset(s,r);}
void CudaRealtimeRenderer::render_next_frame_to_surface(const RenderSceneSnapshot& s,const Camera& c,const RenderSettings& r,const InteractiveFrameState& f,CudaSurfaceHandle target){impl_->render(s,c,r,f,target);}
void CudaRealtimeRenderer::render_next_frame(const RenderSceneSnapshot& s,const Camera& c,const RenderSettings& r,const InteractiveFrameState& f,Framebuffer& target){impl_->render(s,c,r,f,0);impl_->download(target);}
void CudaRealtimeRenderer::download_current_frame(Framebuffer& target){impl_->download(target);}
std::vector<RealtimeDiagnosticPixel> CudaRealtimeRenderer::download_diagnostics(){return impl_->diagnostics();}
CudaStreamHandle CudaRealtimeRenderer::stream_handle() const{return reinterpret_cast<CudaStreamHandle>(impl_->stream_);}
const CudaPathStatistics& CudaRealtimeRenderer::statistics() const{return impl_->statistics_;}
void CudaRealtimeRenderer::refresh_statistics(){impl_->refresh();}
void CudaRealtimeRenderer::set_presentation_state(bool a,bool b){impl_->statistics_.interop_active=a;impl_->statistics_.fallback_active=b;}

} // namespace renderer
