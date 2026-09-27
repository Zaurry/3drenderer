#pragma once
#include "render/realtime/sobol_directions.cuh"

// Only RTRT uses this sampler; the offline renderer retains its PCG stream.
// Distinct Sobol dimensions avoid the correlations of reusing one radical
// inverse for all random choices. Pixel scrambling stays fixed across frames.
struct RtSampler {
    DPcgState fallback{};
    unsigned int index=0, scramble=0, dimension=0;
    bool low_discrepancy=false;
};
__device__ unsigned int rt_hash(unsigned int v) {
    v^=v>>16;v*=0x7feb352du;v^=v>>15;v*=0x846ca68bu;return v^(v>>16);
}
__device__ float random_float(RtSampler& rng) {
    if(!rng.low_discrepancy || rng.dimension>=16) return random_float(rng.fallback);
    const unsigned int dimension=rng.dimension++;
    unsigned int value=0,index=rng.index;
    for(int bit=0;index;index>>=1,++bit) if(index&1) value^=rt_sobol_directions[dimension][bit];
    const unsigned int seed=rt_hash(rng.scramble+dimension*0x9e3779b9u);
    // Adapted from pbrt-v4's FastOwenScrambler (Apache-2.0), Copyright(c)
    // 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys. Uses CUDA bit
    // reversal and a 24-bit float conversion. License: PBRT_LICENSE.txt.
    value=__brev(value);value^=value*0x3d20adeau;value+=seed;
    value*=(seed>>16)|1u;value^=value*0x05526c56u;value^=value*0x53a22864u;
    return float(__brev(value)>>8)*(1.0f/16777216.0f);
}

__device__ void rt_gbuffer_pixel(RtFrame f, RtGuide* guides, DCompactHit* primary_hits,int i,DVec3* emission=nullptr) {
    if(i>=f.width*f.height) return;
    RtGuide g{};
    const DRay ray=rt_primary(f,i);
    DCompactHit compact{}; compact.primitive_kind=-1;
    if(rt_intersect(f,ray,1e30f,compact)) {
        DHit hit{}; reconstruct_hit(f.scene,ray,compact,true,hit);
        DMaterial m{};
        if (hit.material_id>=0 && hit.material_id<f.scene.material_count) m=f.scene.materials[hit.material_id];
        else { m.type=int(MaterialType::Diffuse); m.base_color=v3(1,0,1); m.roughness=1; }
        const DSurface s=evaluate_surface(f.scene,m,hit);
        if(emission) emission[i]=hit.material_id>=0 && hit.material_id<f.scene.material_count?s.emission:v3(1,0,1);
        g.normal=s.shading_normal; g.geometric=rt_pack_normal(hit.geometric_normal);
        g.albedo=s.diffuse_color; g.roughness=s.roughness;
        g.material=hit.material_id;
        g.transparent=(m.type==int(MaterialType::Dielectric) || effective_alpha_mode(m)==int(AlphaMode::Blend));
        DVec3 previous_position=hit.position;
        g.previous_normal=g.normal;
        g.previous_geometric=g.geometric;
        if(hit.instance_index>=0) {
            const RtInstance instance=f.instances[hit.instance_index];
            g.object_id=instance.object_id; g.asset_id=instance.asset_id;
            previous_position=transform_point(instance.current_to_previous,hit.position);
            g.previous_normal=normalize(transform_direction(instance.normal_to_previous,g.normal));
            g.previous_geometric=rt_pack_normal(normalize(transform_direction(instance.normal_to_previous,hit.geometric_normal)));
        }
        const DVec2 uv=rt_project(f.camera,hit.position,g.depth);
        const DVec2 old=rt_project(f.previous_camera,previous_position,g.previous_depth);
        g.motion={old.x-uv.x,old.y-uv.y};
    } else {
        compact.primitive_kind=-1;
        if(emission) emission[i]=f.scene.environment_background_visible?environment_radiance(f.scene,ray.direction):v3(0,0,0);
        // Sky has rotational flow and no finite depth. Never reconstruct it as a surface.
        float z=0,old_z=0;
        const DVec2 uv=rt_project(f.camera,add(f.camera.eye,ray.direction),z);
        const DVec2 old=rt_project(f.previous_camera,add(f.previous_camera.eye,ray.direction),old_z);
        g.motion={old.x-uv.x,old.y-uv.y}; g.previous_depth=old_z>0?0:-1;
    }
    guides[i]=g; if(primary_hits)primary_hits[i]=compact;
}
__device__ DVec3 rt_cone(DVec3 direction,float angle,RtSampler& rng) {
    if(angle<=0) return direction;
    const float z=1-random_float(rng)*(1-cosf(angle));
    const float phi=2*kPi*random_float(rng);
    const float r=sqrtf(fmaxf(0,1-z*z));
    return tangent_to_world(v3(r*cosf(phi),r*sinf(phi),z),direction);
}
__device__ bool rt_visible(const RtFrame& f,const DHit& hit,DVec3 direction,float distance,bool casts) {
    if(!f.settings.shadows || !casts) return true;
    const DVec3 origin=offset_origin(hit.position,hit.geometric_normal,direction);
    return !rt_occluded(f,{origin,direction},distance<1e29f?fmaxf(0,distance*(1-1e-5f)):distance);
}
struct RtDirect { DVec3 diffuse{}, specular{}; };
__device__ void rt_add_direct(RtDirect& out,const DSurface& surface,DVec3 outgoing,DVec3 incoming,DVec3 radiance) {
    const DPbrEvaluation e=evaluate_pbr(surface,surface.shading_normal,outgoing,incoming);
    const DVec3 incoming_cos=mul(radiance,fmaxf(0,dot(surface.shading_normal,incoming)));
    out.diffuse=add(out.diffuse,product(e.diffuse,incoming_cos));
    out.specular=add(out.specular,product(e.specular,incoming_cos));
}
__device__ RtDirect rt_direct(const RtFrame& f,const DHit& hit,const DSurface& s,DVec3 outgoing,RtSampler& rng,bool bsdf_sample) {
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
        DShadowTask task{};DVec3 fraction{};
        const float ep=environment_strategy_probability(f.scene);
        const bool environment=ep>0 && (ep>=1 || random_float(rng)<ep);
        const bool sampled=environment
            ? sample_environment_shadow_task(f.scene,hit,s,outgoing,v3(1,1,1),0,rng,task,bsdf_sample,&fraction)
            : sample_emissive_shadow_task(f.scene,hit,s,outgoing,v3(1,1,1),0,rng,task,bsdf_sample,&fraction);
        if(sampled && (!f.settings.shadows || !task.casts_shadows || !rt_occluded(f,task.ray,task.t_max))) {
            // Reuse the BRDF split evaluated for the actual light direction;
            // the offset visibility ray can have a slightly different one.
            const DVec3 contribution=mul(task.contribution,1.0f/samples);
            const DVec3 diffuse=product(contribution,fraction);
            out.diffuse=add(out.diffuse,diffuse);
            out.specular=add(out.specular,sub(contribution,diffuse));
        }
    }
    return out;
}

__device__ RtSignals rt_trace_hit(RtFrame f,const DCompactHit& primary,RtGuide& guide,int i) {
    RtSignals output{}; float distances=0; int distance_samples=0;
    const int primary_material=guide.material;
    const bool split_glass=f.settings.split_dielectric && guide.depth>0 && primary_material>=0 && primary_material<f.scene.material_count &&
        f.scene.materials[primary_material].type==int(MaterialType::Dielectric) &&
        effective_alpha_mode(f.scene.materials[primary_material])!=int(AlphaMode::Blend);
    const int branches=split_glass?3:1;
    for(int sample=0;sample<f.settings.samples_per_pixel;++sample) for(int branch=0;branch<branches;++branch) {
        RtSampler rng{};
        pcg_seed(rng.fallback,pixel_seed(i%f.width,i/f.width,f.width,f.seed+f.frame_index*0x9e3779b97f4a7c15ULL+sample*0x85ebca6bULL));
        const auto pixel=pixel_seed(i%f.width,i/f.width,f.width,f.seed);
        rng.scramble=rt_hash(unsigned(pixel)^unsigned(pixel>>32));
        rng.index=unsigned(((f.frame_index-1)*f.settings.samples_per_pixel+sample)*branches+branch);
        rng.low_discrepancy=f.settings.low_discrepancy;
        DRay ray=rt_primary(f,i);
        DVec3 throughput=v3(1,1,1),wd{},ws{},wt{};
        float previous_pdf=0; int previous_delta=1;
        bool split_exit=false,dielectric_chain=split_glass,regularize_path=false;
        for(int bounce=0;bounce<f.settings.max_bounces;++bounce) {
            DCompactHit compact{};
            bool found=false;
            if(bounce==0) { compact=primary; found=compact.primitive_kind>=0; }
            else found=rt_intersect(f,ray,1e30f,compact,true);
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
            DSurface surface=evaluate_surface(f.scene,material,hit);
            // The primary footprint filter must affect evaluation, sampling and
            // their PDFs together; changing only the denoiser guide is biased.
            if(bounce==0) {surface.roughness=guide.roughness;surface.shading_normal=guide.normal;}
            // Once a broad scattering event has destroyed directional detail,
            // an almost-delta secondary lobe produces rare, unresolvable paths.
            // Regularize only downstream of that event, keeping visible glossy
            // surfaces and uninterrupted mirror/glass chains unchanged.
            if(f.settings.regularize_indirect && regularize_path && material.type!=int(MaterialType::Dielectric)) {
                const float alpha=surface.roughness*surface.roughness;
                if(alpha<.3f)surface.roughness=sqrtf(fminf(.3f,fmaxf(.1f,2*alpha)));
            }
            if(bounce==1) { distances+=hit.t; ++distance_samples; }
            const bool passthrough=effective_alpha_mode(material)==int(AlphaMode::Blend) && random_float(rng)>=surface.opacity;
            if(!passthrough && max_component(surface.emission)>0) {
                const float weight=previous_delta?1:power_heuristic(previous_pdf,emissive_light_pdf_for_hit(f.scene,ray.origin,hit));
                const DVec3 incoming=mul(surface.emission,weight);
                if(bounce==0) {if(branch==0)output.c[4]=add(output.c[4],incoming);}
                else { output.c[1]=add(output.c[1],product(wd,incoming)); output.c[2]=add(output.c[2],product(ws,incoming)); output.c[3]=add(output.c[3],product(wt,incoming)); }
            }
            if(!passthrough && material.type==int(MaterialType::Emissive)) break;
            // Match scatter()'s normalized view for direct BRDF/PDF evaluation;
            // one ULP in N.H is significant for nearly delta GGX surfaces.
            const DVec3 outgoing=normalize(mul(ray.direction,-1));
            if(!passthrough && material.type!=int(MaterialType::Dielectric) && (bounce>0 || f.settings.direct_lighting)) {
                const RtDirect d=rt_direct(f,hit,surface,outgoing,rng,bounce+1<f.settings.max_bounces);
                if(bounce==0) {
                    output.c[0]=add(output.c[0],d.diffuse); output.c[2]=add(output.c[2],d.specular);
                    output.direct=add(output.direct,add(d.diffuse,d.specular));
                } else {
                    const DVec3 incoming=add(d.diffuse,d.specular);
                    output.c[1]=add(output.c[1],product(wd,incoming)); output.c[2]=add(output.c[2],product(ws,incoming)); output.c[3]=add(output.c[3],product(wt,incoming));
                }
            }
            // No continuation ray is consumed after the final allowed vertex.
            if(bounce+1>=f.settings.max_bounces) break;
            DRay scattered{}; DVec3 attenuation{},diffuse_fraction{}; float pdf=0; int delta=1;
            if(passthrough) { scattered={offset_origin(hit.position,hit.geometric_normal,ray.direction),ray.direction}; attenuation=v3(1,1,1); }
            else if(split_glass && material.type==int(MaterialType::Dielectric) &&
                (bounce==0 || (branch>0 && !split_exit))) {
                const float ratio=hit.front_face?1/material.ior:material.ior;
                const DVec3 incoming=normalize(ray.direction);
                const float cosine=saturate(dot(mul(incoming,-1),surface.shading_normal));
                DVec3 refracted{};
                const bool can_refract=refract_vector(incoming,surface.shading_normal,ratio,refracted);
                const float fresnel=can_refract?reflectance(cosine,ratio):1;
                const bool reflection=bounce==0?branch==0:branch==1;
                // One reflected path plus two half-weight transmitted paths.
                // The latter stratify the next dielectric event. If no second
                // interface is hit, their half weights still sum correctly.
                const float weight=bounce==0?(reflection?fresnel:.5f*(1-fresnel)):
                    2*(reflection?fresnel:1-fresnel);
                if(weight<=0)break;
                const DVec3 direction=normalize(reflection?reflect_vector(incoming,surface.shading_normal):refracted);
                scattered={offset_origin(hit.position,hit.geometric_normal,direction),direction};
                attenuation=v3(weight,weight,weight);
                if(bounce>0)split_exit=true;
            }
            else if(!scatter(ray,hit,material,surface,rng,attenuation,scattered,pdf,delta,bounce==0?&diffuse_fraction:nullptr)) break;
            dielectric_chain=dielectric_chain && material.type==int(MaterialType::Dielectric);
            if(bounce==0) {
                if(passthrough || material.type==int(MaterialType::Dielectric)) {
                    // Delta transmission and reflection keep their own short history.
                    const bool transmitted=passthrough || dot(scattered.direction,hit.geometric_normal)<0;
                    if(transmitted) wt=f.settings.transmission?attenuation:v3(0,0,0);
                    else ws=f.settings.reflections?attenuation:v3(0,0,0);
                } else {
                    // Reuse the evaluation that produced the sampling PDF.
                    // Besides saving a BRDF evaluation, this avoids amplifying
                    // normalization roundoff in almost-delta specular lobes.
                    const DVec3 diffuse=product(attenuation,diffuse_fraction);
                    wd=f.settings.indirect_diffuse?diffuse:v3(0,0,0);
                    ws=f.settings.reflections?rt_safe(sub(attenuation,diffuse)):v3(0,0,0);
                }
            } else { wd=product(wd,attenuation); ws=product(ws,attenuation); wt=product(wt,attenuation); }
            if(!delta && (surface.roughness>=.2f || (bounce==0 && rt_luma(wd)>rt_luma(ws))))regularize_path=true;
            throughput=add(add(wd,ws),wt);
            if(!finite(throughput) || max_component(throughput)<=0) break;
            // Roulette on a weak but deterministic internal glass reflection
            // would turn it back into a rare, high-energy white sample.
            if(bounce+1>=f.settings.roulette_start && !dielectric_chain) {
                const float p=fminf(.95f,fmaxf(.05f,max_component(throughput)));
                if(random_float(rng)>=p) break;
                wd=divv(wd,p); ws=divv(ws,p); wt=divv(wt,p);
            }
            ray=scattered; previous_pdf=pdf; previous_delta=delta || (bounce==0 && !f.settings.direct_lighting);
        }
    }
    for(int c=0;c<5;++c) output.c[c]=rt_safe(mul(output.c[c],1.0f/f.settings.samples_per_pixel));
    output.direct=rt_safe(mul(output.direct,1.0f/f.settings.samples_per_pixel));
    guide.hit_distance=distance_samples>0?distances/distance_samples:0;
    return output;
}
__device__ void rt_trace_pixel(RtFrame f,const DCompactHit* primary_hits,RtGuide* guides,RtSignals* signals,int i) {
    if(i<f.width*f.height)signals[i]=rt_trace_hit(f,primary_hits[i],guides[i],i);
}
__device__ void rt_native_optics_pixel(RtFrame f,const DCompactHit* primary_hits,RtGuide* guides,DVec3* emission,int i) {
    if(i>=f.width*f.height || !rt_sharp_optics(guides[i]))return;
    // Reuse the native primary hit. A separate launch keeps path-tracing
    // register pressure out of the inexpensive primary visibility pass.
    const RtSignals lighting=rt_trace_hit(f,primary_hits[i],guides[i],i);
    DVec3 color{};for(int c=0;c<5;++c)color=add(color,lighting.c[c]);
    emission[i]=rt_safe(color);
}


