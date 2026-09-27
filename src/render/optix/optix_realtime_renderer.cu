#include "render/optix/realtime_device.cuh"
#include "render/optix/optix_backend.h"
#include "render/optix/optix_denoiser.h"

namespace renderer {
namespace {
__global__ void rt_compact_optics(const RtGuide* guides,int pixels,int* indices,unsigned* count) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x),lane=int(threadIdx.x)&31;
    const bool selected=i<pixels && rt_sharp_optics(guides[i]);
    const unsigned mask=__ballot_sync(0xffffffffu,selected);
    if(!mask)return;
    const int leader=__ffs(mask)-1;unsigned base=0;
    if(lane==leader)base=atomicAdd(count,unsigned(__popc(mask)));
    base=__shfl_sync(0xffffffffu,base,leader);
    if(selected)indices[base+__popc(mask&((1u<<lane)-1))]=i;
}
template<class A,class B>
__device__ bool rt_same_surface(const A& a,const B& b,float normal_limit=.8f) {
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
__device__ DVec3 rt_demodulate(DVec3 color,const RtGuide& guide,int channel) {
    return channel<2?rt_divide(color,rt_albedo(guide)):color;
}
__device__ DVec3 rt_modulate(DVec3 color,const RtGuide& guide,int channel) {
    return channel<2?product(color,rt_albedo(guide)):color;
}

struct RtSurfaceFilter {DVec3 normal;float roughness;};
__global__ void rt_filter_specular_surface(RtFrame f,const RtGuide* guides,RtSurfaceFilter* filtered) {
    constexpr int stride=kPrepareWidth+2,rows=kPrepareHeight+2,pixels=stride*rows;
    __shared__ RtNeighborhoodGuide tile[pixels];
    __shared__ unsigned geometric_normals[pixels];
    const int tx=int(threadIdx.x),ty=int(threadIdx.y);
    const int origin_x=int(blockIdx.x)*kPrepareWidth,origin_y=int(blockIdx.y)*kPrepareHeight;
    // Share the footprint and its depth-gradient samples; fetching the full
    // interleaved guide for every neighbor is expensive at native resolution.
    for(int p=ty*kPrepareWidth+tx;p<pixels;p+=kPrepareWidth*kPrepareHeight) {
        const int x=min(f.width-1,max(0,origin_x+p%stride-1));
        const int y=min(f.height-1,max(0,origin_y+p/stride-1));
        const RtGuide g=guides[y*f.width+x];
        tile[p]={g.normal,g.depth,g.object_id,g.asset_id,g.material};
        geometric_normals[p]=g.geometric;
    }
    __syncthreads();
    const int ix=origin_x+tx,iy=origin_y+ty;
    if(ix>=f.width || iy>=f.height)return;
    const int i=iy*f.width+ix,center=(ty+1)*stride+tx+1;
    const RtGuide g=guides[i];
    filtered[i]={g.normal,g.roughness};
    if(g.depth<=0 || g.transparent || g.material<0 || g.material>=f.scene.material_count)return;
    const DMaterial material=f.scene.materials[g.material];
    if(material.normal_texture_id<0 && material.bump_texture_id<0)return;
    const DVec3 geometric=rt_unpack_normal(g.geometric);
    float dx=1e20f,dy=1e20f;
    if(ix>0 && rt_same_surface(g,tile[center-1]))dx=fminf(dx,fabsf(g.depth-tile[center-1].depth));
    if(ix+1<f.width && rt_same_surface(g,tile[center+1]))dx=fminf(dx,fabsf(g.depth-tile[center+1].depth));
    if(iy>0 && rt_same_surface(g,tile[center-stride]))dy=fminf(dy,fabsf(g.depth-tile[center-stride].depth));
    if(iy+1<f.height && rt_same_surface(g,tile[center+stride]))dy=fminf(dy,fabsf(g.depth-tile[center+stride].depth));
    const float gradient=(dx<1e19f?dx:0)+(dy<1e19f?dy:0);
    DVec3 mean=mul(g.normal,4);float weight=4;
    for(int y=-1;y<=1;++y)for(int x=-1;x<=1;++x) {
        if((x==0 && y==0) || ix+x<0 || iy+y<0 || ix+x>=f.width || iy+y>=f.height)continue;
        const int p=center+y*stride+x;const auto& q=tile[p];
        if(q.depth<=0 || g.object_id!=q.object_id || g.asset_id!=q.asset_id || g.material!=q.material ||
            dot(geometric,rt_unpack_normal(geometric_normals[p]))<.95f ||
            fabsf(g.depth-q.depth)>.02f*g.depth+gradient*float(abs(x)+abs(y)+1))continue;
        const float w=x==0 || y==0?2.0f:1.0f;
        mean=add(mean,mul(q.normal,w));weight+=w;
    }
    mean=divv(mean,weight);
    if(!usable(mean) || dot(mean,geometric)<=0)return;
    // Approximate the footprint's normal distribution by a vMF lobe, then
    // convolve its variance with the BRDF (Toksvig / normal-map filtering).
    // 1/(2*kappa), kappa=(3L-L^3)/(1-L^2), preserves the GGX roughness scale.
    const float length2=fminf(1,dot(mean,mean)),length=sqrtf(length2);
    const float variance=(1-length2)/fmaxf(2*length*(3-length2),1e-6f);
    const float alpha=g.roughness*g.roughness;
    filtered[i]={normalize(mean),sqrtf(sqrtf(fminf(1,alpha*alpha+fminf(variance,.0625f))))};
}
__global__ void rt_apply_specular_surface(RtFrame f,const DCompactHit* hits,RtGuide* guides,const RtSurfaceFilter* filtered) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i>=f.width*f.height)return;
    RtGuide& g=guides[i];const RtSurfaceFilter value=filtered[i];
    g.normal=value.normal;g.roughness=value.roughness;
    g.previous_normal=value.normal;
    const int instance=hits[i].instance_index;
    if(g.depth>0 && instance>=0)
        g.previous_normal=normalize(transform_direction(f.instances[instance].normal_to_previous,value.normal));
}

__device__ __forceinline__ DVec3 rt_denoising_signal(bool deterministic_direct,const RtSignals& raw,const RtGuide& g,int channel) {
    if(deterministic_direct && !g.transparent) {
        // Keep direct-light history for change detection, while presenting its
        // current deterministic value without spatial filtering.
        if(channel==0)return raw.direct;
        if(channel==2)return rt_safe(sub(raw.c[2],sub(raw.direct,raw.c[0])));
    }
    return raw.c[channel];
}

__device__ bool rt_valid_history(const RtFrame&,const RtGuide&,const RtGuide&,float);
__device__ DVec2 rt_history_pixel(const RtFrame&,int,const RtGuide&);

__global__ void rt_prepare_signal(RtFrame f,const RtGuide* guides,const RtSignals* raw,RtPrepared* prepared,
    const RtGuide* previous_guides,const RtHistory* previous_history) {
    // Demodulate each halo sample once, then share geometry and luminance for
    // firefly suppression AND temporal neighborhood statistics (17 KiB/block).
    __shared__ RtNeighborhoodGuide tile_guides[kPreparePixels];
    __shared__ RtLuminance tile_luma[kPreparePixels];
    const int tx=int(threadIdx.x),ty=int(threadIdx.y);
    const int origin_x=int(blockIdx.x)*kPrepareWidth,origin_y=int(blockIdx.y)*kPrepareHeight;
    for(int p=ty*kPrepareWidth+tx;p<kPreparePixels;p+=kPrepareWidth*kPrepareHeight) {
        const int x=min(f.width-1,max(0,origin_x+p%kPrepareStride-kPrepareRadius));
        const int y=min(f.height-1,max(0,origin_y+p/kPrepareStride-kPrepareRadius));
        const int q=y*f.width+x; const RtGuide g=guides[q];
        tile_guides[p]={g.normal,g.depth,g.object_id,g.asset_id,g.material};
        #pragma unroll
        for(int c=0;c<kSignals;++c) tile_luma[p].c[c]=rt_luma(rt_demodulate(rt_denoising_signal(f.deterministic_direct,raw[q],g,c),g,c));
    }
    __syncthreads();
    const int ix=origin_x+tx,iy=origin_y+ty;
    if(ix>=f.width || iy>=f.height) return;
    const int i=iy*f.width+ix;
    RtPrepared result{};
    const RtGuide g=guides[i];
    // Background radiance lives in the unfiltered emission signal. All lanes
    // must finish the cooperative tile load before this early return.
    if(g.depth<=0) {prepared[i]=result;return;}
    const float gradient=rt_gradient(guides,i,f.width,f.height);
    float sum[kSignals]{},sum2[kSignals]{},largest[kSignals]{},second[kSignals]{},count=0;
    if(f.settings.denoise) {
        for(int y=-3;y<=3;++y) for(int x=-3;x<=3;++x) {
            const int qx=ix+x,qy=iy+y;
            if(qx<0 || qx>=f.width || qy<0 || qy>=f.height) continue;
            if(x==0 && y==0) continue;
            const int q=(ty+y+kPrepareRadius)*kPrepareStride+tx+x+kPrepareRadius;
            const auto neighbor=tile_guides[q];
            const float tolerance=.02f*fmaxf(g.depth,1e-3f)+gradient*float(abs(x)+abs(y)+1);
            if(!rt_same_surface(g,neighbor) || fabsf(g.depth-neighbor.depth)>tolerance) continue;
            #pragma unroll
            for(int c=0;c<kSignals;++c) {
                const float l=tile_luma[q].c[c];
                // Accumulate without the two largest samples so subtraction
                // of an extreme value cannot erase the remaining precision.
                const float accepted=fminf(second[c],l);
                sum[c]+=accepted;sum2[c]+=accepted*accepted;
                second[c]=fmaxf(second[c],fminf(largest[c],l));largest[c]=fmaxf(largest[c],l);
            }
            count+=1;
        }
    }
    #pragma unroll
    for(int c=0;c<kSignals;++c) {
        DVec3 color=rt_demodulate(rt_denoising_signal(f.deterministic_direct,raw[i],g,c),g,c);
        const bool sharp=c>=2 && (g.transparent || g.roughness<.06f);
        const float total=sum[c]+largest[c]+second[c];
        const float total2=sum2[c]+largest[c]*largest[c]+second[c]*second[c];
        float neighborhood_sum=total,neighborhood_sum2=total2;
        if(f.settings.firefly_filter && count>=8 && (c<3 || sharp)) {
            // A second hot sample must not inflate the center's rejection
            // threshold. Trim the upper tail before estimating that threshold,
            // then winsorize these samples in the moments passed to history.
            const float included=count>=16?0:second[c];
            const float n=count-(count>=16?2:1);
            const float mean=(sum[c]+included)/n;
            const float variance=fmaxf(0,(sum2[c]+included*included)/n-mean*mean);
            float limit=mean+f.settings.firefly_sigma*sqrtf(variance)+.02f;
            const float l=rt_luma(color);
            if(sharp && l>limit) {
                // Preserve established sharp highlights. With no trustworthy
                // history, or a changed secondary hit, let TAA resolve coverage
                // instead of mistaking a newly visible reflection for noise.
                const DVec2 hp=rt_history_pixel(f,i,g);
                const int x=int(floorf(hp.x+.5f)),y=int(floorf(hp.y+.5f));
                bool supported=false;
                if(f.valid_history && !f.shading_changed && x>=0 && y>=0 && x<f.width && y<f.height) {
                    const int q=y*f.width+x;const RtGuide old=previous_guides[q];
                    const float tolerance=f.settings.depth_threshold*fmaxf(g.previous_depth,1e-3f)+2*gradient;
                    const bool distance_matches=(g.hit_distance>0)==(old.hit_distance>0) &&
                        fabsf(g.hit_distance-old.hit_distance)<=.1f*fmaxf(g.hit_distance,.001f);
                    if(rt_valid_history(f,g,old,tolerance) && distance_matches && previous_history[q].length[c]>=4) {
                        limit=fmaxf(limit,2*rt_luma(previous_history[q].color[c])+.02f);supported=true;
                    }
                }
                if(!supported)limit=l;
            }
            if(l>limit && l>1e-8f) color=mul(color,limit/l);
            const float a=fminf(largest[c],limit),b=fminf(second[c],limit);
            neighborhood_sum=sum[c]+a+b;neighborhood_sum2=sum2[c]+a*a+b*b;
        }
        result.c[c]=color;
        const float l=rt_luma(color);
        result.mean[c]=(neighborhood_sum+l)/(count+1);
        result.variance[c]=fmaxf(0,(neighborhood_sum2+l*l)/(count+1)-result.mean[c]*result.mean[c]);
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
    const RtPrepared* prepared,const RtHistory* previous,RtHistory* history,
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
    float reactive_sum=0,reactive_weight=0;
    for(int c=0;c<kSignals;++c) {
        const float mean=prepared[i].mean[c],spatial_variance=prepared[i].variance[c];
        const DVec3 current=prepared[i].c[c]; const float l=rt_luma(current);
        float alpha=1,reactive=0,length=1,history_energy=0; DVec3 old_color{}; DVec2 old_moments{};
        if(sumw>1e-5f) {
            float old_length=0;
            for(int tap=0;tap<taps;++tap) {
                const float w=weights[tap]/sumw; const int q=indices[tap];
                old_color=add(old_color,mul(previous[q].color[c],w));
                old_moments.x+=previous[q].moments[c].x*w; old_moments.y+=previous[q].moments[c].y*w;
                old_length+=previous[q].length[c]*w;
            }
            const float old_l=rt_luma(old_color);
            history_energy=rt_luma(rt_modulate(old_color,g,c));
            // A sparse 1-SPP neighborhood can be dark by chance. Its variance
            // alone must not erase a converged history and restart accumulation.
            const float old_variance=fmaxf(0,old_moments.y-old_moments.x*old_moments.x);
            const float sigma=sqrtf(f.shading_changed?spatial_variance:fmaxf(spatial_variance,old_variance));
            reactive=saturate((fabsf(old_l-mean)/(3*sigma+.15f*fabsf(mean)+.02f)-1)*f.settings.reactive_strength);
            if(f.shading_changed) reactive=fmaxf(reactive,.5f);
            if(f.settings.history_clamping && old_l>1e-8f) {
                // Neighboring mirror/refraction pixels can legitimately see
                // completely different radiance. A neighborhood lower bound
                // must not manufacture light in their dark histories.
                const bool sharp=c>=2 && (g.transparent || g.roughness<.06f);
                const float lower=sharp?0:fmaxf(0,mean-f.settings.history_sigma*sigma);
                const float target=fminf(fmaxf(old_l,lower),mean+f.settings.history_sigma*sigma+.001f);
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
        // Sharp optical detail follows secondary geometry, not this primary
        // surface's motion. Repeated bilinear SVGF history would spread a thin
        // highlight and bias its energy. Accumulate it once, in output TAA.
        t.c[c]=c>=2 && (g.transparent || g.roughness<.06f)?current:h.color[c];
        const float v=fmaxf(0,h.moments[c].y-h.moments[c].x*h.moments[c].x);
        t.variance[c]=length<4?fmaxf(v,spatial_variance)*4/fmaxf(length,1):v;
        // Include disappearing radiance so a newly dark shadow still reacts.
        const float energy=fmaxf(history_energy,rt_luma(rt_modulate(current,g,c)));
        reactive_sum+=reactive*energy; reactive_weight+=energy;
        diagnostic.variance=fmaxf(diagnostic.variance,t.variance[c]);
    }
    // Do not let an empty/noisy minor component discard the complete TAA image.
    diagnostic.reactive=reactive_weight>1e-8f?reactive_sum/reactive_weight:0;
    if(f.shading_changed) diagnostic.reactive=fmaxf(diagnostic.reactive,.5f);
    diagnostic.history=h.length[1]; history[i]=h; temporal[i]=t; diagnostics[i]=diagnostic;
}

struct RtAtrousGuide {
    DVec3 normal;float depth;
    unsigned long long object_id,asset_id;
    int material;float roughness;
};
__device__ RtAtrousGuide rt_atrous_guide(const RtGuide& g) {
    return {g.normal,g.depth,g.object_id,g.asset_id,g.material,g.roughness};
}
struct RtAtrousSample {
    DVec3 c[3];float variance[3];
    // A 13-float stride avoids the 16-way bank conflicts of RtFiltered (64 B).
    float padding;
};
static_assert(sizeof(RtAtrousSample)==13*sizeof(float));
__device__ RtAtrousSample rt_atrous_sample(const RtFiltered& s) {
    RtAtrousSample result{};
    #pragma unroll
    for(int c=0;c<3;++c) {result.c[c]=s.c[c];result.variance[c]=s.variance[c];}
    return result;
}

// The two dense 5x5 passes share their halo instead of reloading it for every
// pixel and signal. Coarse, sparse passes keep direct global reads. Keep the
// stencil, accumulation order and first-pass history feedback identical.
template<int FineIteration=-1>
__global__ void rt_atrous(RtFrame f,const RtGuide* guides,const RtFiltered* input,RtFiltered* output,
    RtHistory* feedback,int iteration) {
    static_assert(FineIteration>=-1 && FineIteration<=1);
    constexpr int halo=FineIteration==0?2:(FineIteration==1?4:0);
    constexpr int stride=kPrepareWidth+2*halo;
    constexpr int tile_size=FineIteration>=0?stride*(kPrepareHeight+2*halo):1;
    __shared__ RtAtrousGuide tile_guides[tile_size];
    __shared__ RtAtrousSample tile_signals[tile_size];
    int ix,iy,tile_center=0;
    if constexpr(FineIteration>=0) {
        const int tx=int(threadIdx.x),ty=int(threadIdx.y);
        const int ox=int(blockIdx.x)*kPrepareWidth,oy=int(blockIdx.y)*kPrepareHeight;
        for(int p=ty*kPrepareWidth+tx;p<tile_size;p+=kPrepareWidth*kPrepareHeight) {
            const int x=min(f.width-1,max(0,ox+p%stride-halo));
            const int y=min(f.height-1,max(0,oy+p/stride-halo));
            const int q=y*f.width+x;
            tile_guides[p]=rt_atrous_guide(guides[q]);tile_signals[p]=rt_atrous_sample(input[q]);
        }
        __syncthreads();
        ix=ox+tx;iy=oy+ty;tile_center=(ty+halo)*stride+tx+halo;
        iteration=FineIteration;
    } else {
        const int index=int(blockIdx.x*blockDim.x+threadIdx.x);
        ix=index%f.width;iy=index/f.width;
    }
    if(ix>=f.width || iy>=f.height) return;
    const int i=iy*f.width+ix;
    const int step=1<<iteration; const RtGuide g=guides[i]; RtFiltered result{};
    if(g.depth<=0) {
        output[i]=input[i];
        if(iteration==0) for(int c=0;c<kSignals;++c) feedback[i].color[c]=input[i].c[c];
        return;
    }
    const float gradient=rt_gradient(guides,i,f.width,f.height);
    const float kernel[5]={1,4,6,4,1};
    bool active[kSignals]; float lp[kSignals],sigma[kSignals];
    DVec3 sum[kSignals]{}; float weights[kSignals]{},variance[kSignals]{};
    bool any_active=false;
    // Preblur variance to keep the edge stopper from tracing individual noise
    // samples as the propagated variance contracts over successive iterations.
    float local_variance[kSignals]{},local_weight=0;
    for(int y=-1;y<=1;++y) for(int x=-1;x<=1;++x) {
        const int qx=i%f.width+x,qy=i/f.width+y;
        if(qx<0 || qy<0 || qx>=f.width || qy>=f.height) continue;
        const int q=qy*f.width+qx;
        const auto neighbor=FineIteration>=0?tile_guides[tile_center+y*stride+x]:rt_atrous_guide(guides[q]);
        const float tolerance=.02f*fmaxf(g.depth,1e-3f)+gradient*float(abs(x)+abs(y)+1);
        if(!rt_same_surface(g,neighbor) || fabsf(g.depth-neighbor.depth)>tolerance) continue;
        const auto sample=FineIteration>=0?tile_signals[tile_center+y*stride+x]:rt_atrous_sample(input[q]);
        const float w=float((x==0?2:1)*(y==0?2:1));
        #pragma unroll
        for(int c=0;c<3;++c) local_variance[c]+=w*sample.variance[c];
        local_weight+=w;
    }
    #pragma unroll
    for(int c=0;c<kSignals;++c) {
        const int iterations=c<2?f.settings.diffuse_iterations:(c==2?f.settings.specular_iterations:0);
        const bool delta=(c>=2 && (g.transparent || g.roughness<.06f));
        active[c]=f.settings.denoise && iteration<iterations && !delta && g.depth>0 &&
            !(c==0 && f.deterministic_direct && !g.transparent);
        any_active=any_active || active[c];
        result.c[c]=input[i].c[c]; result.variance[c]=input[i].variance[c];
        lp[c]=rt_luma(input[i].c[c]);
        sigma[c]=f.settings.luminance_sigma*sqrtf(fmaxf(fmaxf(input[i].variance[c],local_variance[c]/fmaxf(local_weight,1)),1e-8f))+.001f;
    }
    if(any_active) {
        // Keep the fine 5x5 passes (including temporal feedback). At coarse
        // strides a 3x3 stencil reduces bandwidth and excessive filter reach.
        const int radius=iteration<2?2:1;
        for(int y=-radius;y<=radius;++y) for(int x=-radius;x<=radius;++x) {
            const int qx=i%f.width+x*step,qy=i/f.width+y*step;
            if(qx<0 || qx>=f.width || qy<0 || qy>=f.height) continue;
            const int q=qy*f.width+qx;
            const int tq=tile_center+(y*stride+x)*step;
            const auto neighbor=FineIteration>=0?tile_guides[tq]:rt_atrous_guide(guides[q]);
            if(!rt_same_surface(g,neighbor,0)) continue;
            const auto sample=FineIteration>=0?tile_signals[tq]:rt_atrous_sample(input[q]);
            const float depth_scale=f.settings.depth_sigma*(gradient*float(step*(abs(x)+abs(y)))+.002f*fmaxf(g.depth,.001f))+1e-6f;
            // Fetch and evaluate geometry once for all three filtered signals.
            const float spatial=radius==2?kernel[x+2]*kernel[y+2]:float((x==0?2:1)*(y==0?2:1));
            float geometry_weight=spatial*__expf(-fabsf(g.depth-neighbor.depth)/depth_scale);
            geometry_weight*=__powf(fmaxf(0,dot(g.normal,neighbor.normal)),f.settings.normal_power);
            #pragma unroll
            for(int c=0;c<3;++c) {
                if(!active[c]) continue;
                float w=geometry_weight*__expf(-fabsf(lp[c]-rt_luma(sample.c[c]))/sigma[c]);
                if(c==2) w*=__expf(-fabsf(g.roughness-neighbor.roughness)*16);
                sum[c]=add(sum[c],mul(sample.c[c],w));
                variance[c]+=sample.variance[c]*w*w; weights[c]+=w;
            }
        }
    }
    #pragma unroll
    for(int c=0;c<kSignals;++c) {
        if(active[c] && weights[c]>1e-8f) {
            result.c[c]=divv(sum[c],weights[c]);
            result.variance[c]=variance[c]/(weights[c]*weights[c]);
        }
        if(iteration==0) feedback[i].color[c]=result.c[c];
    }
    output[i]=result;
}

__device__ __forceinline__ DVec3 rt_composite(bool deterministic_direct,const RtSignals& raw,const RtFiltered& filtered,const RtGuide& guide) {
    DVec3 color=raw.c[4];
    if(deterministic_direct && !guide.transparent)color=add(color,raw.direct);
    for(int c=0;c<kSignals;++c) {
        if(c==0 && deterministic_direct && !guide.transparent)continue;
        color=add(color,rt_modulate(filtered.c[c],guide,c));
    }
    return rt_safe(color);
}
__global__ void rt_compose(RtFrame f,const RtGuide* guides,const RtSignals* raw,
    const RtFiltered* filtered,DVec3* composed) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i<f.width*f.height) composed[i]=rt_composite(f.deterministic_direct,raw[i],filtered[i],guides[i]);
}

#if RENDERER_HAS_OPTIX
// Preserve the signal decomposition for native material reconstruction and
// diagnostics, without running SVGF before the neural beauty denoiser.
__global__ void rt_optix_signals(RtFrame f,const RtGuide* guides,const RtSignals* raw,const RtPrepared* prepared,
    RtFiltered* signals,RealtimeDiagnosticPixel* diagnostics) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i>=f.width*f.height)return;
    const RtGuide g=guides[i];RtFiltered value{};
    // Direct illumination and resolved reflections contain legitimate isolated
    // highlights. Only regularize rare indirect-diffuse paths before AI; use
    // the prepared moments of all channels to stabilize output history.
    for(int c=0;c<kSignals;++c)value.c[c]=prepared && c==1?prepared[i].c[c]:
        rt_demodulate(rt_denoising_signal(f.deterministic_direct,raw[i],g,c),g,c);
    signals[i]=value;
    RealtimeDiagnosticPixel d{};d.depth=g.depth;d.motion_x=g.motion.x;d.motion_y=g.motion.y;
    if(prepared) {
        float sigma=0;
        for(int c=0;c<kSignals;++c) {
            const float scale=c<2?rt_luma(rt_albedo(g)):1;
            sigma+=scale*sqrtf(fmaxf(0,prepared[i].variance[c]));
        }
        d.variance=sigma*sigma;
    }
    d.reactive=f.shading_changed?1.0f:0.0f;diagnostics[i]=d;
}

__global__ void rt_optix_pack_aovs(int count,const RtGuide* guides,const RtFiltered* signals,DVec3* output) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i>=count)return;
    for(int c=0;c<kSignals;++c)output[std::size_t(c)*count+i]=rt_safe(rt_modulate(signals[i].c[c],guides[i],c));
}

__global__ void rt_optix_unpack_aovs(int count,const RtGuide* guides,const DVec3* aovs,const RtFiltered* raw,RtFiltered* output) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i>=count)return;
    RtFiltered value=raw[i];
    if(guides[i].depth>0)for(int c=0;c<kSignals;++c)
        value.c[c]=rt_demodulate(rt_safe(aovs[std::size_t(c)*count+i]),guides[i],c);
    output[i]=value;
}

__device__ RtOutputGuide rt_denoiser_previous(const RtGuide& g) {
    return {g.depth>0?rt_unpack_normal(g.geometric):v3(0,0,0),g.depth,g.object_id,0,g.material};
}
__device__ RtOutputGuide rt_denoiser_previous(const RtOutputGuide& g) {return g;}

template<class PreviousGuide>
__global__ void rt_optix_guides(RtFrame f,const RtGuide* guides,const PreviousGuide* previous,OptixDenoiserGuides output) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i>=f.width*f.height)return;
    const RtGuide g=guides[i];
    // A diffuse-only albedo is zero on metals/glass; use a neutral guide there.
    const DVec3 albedo=g.depth<=0?v3(0,0,0):(g.transparent || max_component(g.albedo)<.01f?v3(1,1,1):
        v3(saturate(g.albedo.x),saturate(g.albedo.y),saturate(g.albedo.z)));
    reinterpret_cast<DVec3*>(output.albedo)[i]=albedo;
    // AOV models use all three world-space normal components (OptiX 9.1 types).
    reinterpret_cast<DVec3*>(output.normal)[i]=g.depth>0?normalize(g.normal):v3(0,0,0);
    if(!output.flow)return;
    const DVec2 old=rt_history_pixel(f,i,g);
    const bool valid_flow=f.valid_history && g.previous_depth>=0 && isfinite(old.x) && isfinite(old.y) &&
        old.x>-.5f && old.y>-.5f && old.x<f.width-.5f && old.y<f.height-.5f;
    // RTRT stores current-to-previous UVs. OptiX expects the opposite direction
    // in pixel units, including the change in sample jitter on both axes.
    reinterpret_cast<DVec2*>(output.flow)[i]=valid_flow?
        DVec2{float(i%f.width)-old.x,float(i/f.width)-old.y}:DVec2{0,0};
    float trust=0;
    if(valid_flow && !f.denoiser_reset) {
        // Validate the exact jittered grid used by previousOutput, not the
        // unrelated unjittered TAA output (which can also have another size).
        const int bx=int(floorf(old.x)),by=int(floorf(old.y));
        const float fx=old.x-bx,fy=old.y-by;
        const float tolerance=f.settings.depth_threshold*fmaxf(g.previous_depth,1e-3f)+2*rt_gradient(guides,i,f.width,f.height);
        for(int y=0;y<2;++y)for(int x=0;x<2;++x) {
            const int qx=bx+x,qy=by+y;
            if(qx<0 || qy<0 || qx>=f.width || qy>=f.height)continue;
            const RtOutputGuide old_guide=rt_denoiser_previous(previous[qy*f.width+qx]);
            bool valid=(g.depth>0)==(old_guide.depth>0);
            if(g.depth>0)valid=valid && g.object_id==old_guide.object_id && g.material==old_guide.material &&
                dot(rt_unpack_normal(g.previous_geometric),old_guide.normal)>=f.settings.normal_threshold &&
                fabsf(g.previous_depth-old_guide.depth)<=tolerance;
            if(valid)trust+=(x?fx:1-fx)*(y?fy:1-fy);
        }
        // Pixel filtering widens a lobe but does not make first-surface motion
        // valid for the reflected scene. Base confidence on authored roughness
        // as well, otherwise enabling antialiasing introduces neural warping.
        // A diffuse-only material needs no such rejection even if its unused
        // specular roughness happens to be zero.
        if(!f.stationary) {
            bool reflected=false;float roughness=g.roughness;
            if(g.material>=0 && g.material<f.scene.material_count) {
                const DMaterial m=f.scene.materials[g.material];
                reflected=m.type==int(MaterialType::Metal) || m.metallic>0 || m.specular_factor>0;
                roughness=fminf(roughness,m.pbr_workflow==int(PbrWorkflow::SpecularGlossiness)?1-m.glossiness:m.roughness);
            }
            if(g.transparent || (reflected && roughness<.2f))trust=0;
        }
        // Moving objects can alter shadows away from their primary footprint.
        // Retain some matched history but respond promptly to those changes.
        if(f.shading_changed)trust*=.25f;
    }
    output.trustworthiness[i]=trust;
}

__global__ void rt_optix_finish(int count,const RtGuide* guides,const DVec3* noisy,const DVec3* denoised,
    DVec3* result,bool sharp_only,RtOutputGuide* saved_guides) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i>=count)return;
    // previousOutput must remain paired with OptiX's internal guide history.
    // Compositing native materials into that buffer corrupts the next invoke.
    result[i]=guides[i].depth<=0 || (sharp_only && !rt_sharp_optics(guides[i]))?noisy[i]:rt_safe(denoised[i]);
    if(saved_guides)saved_guides[i]=rt_denoiser_previous(guides[i]);
}
#endif
__device__ int rt_guide_index(RtFrame f,int x,int y) {
    return min(f.height-1,max(0,y))*f.width+min(f.width-1,max(0,x));
}

// Reconstruct illumination, then apply materials sampled at output resolution.
// Texture and emission detail consequently do not inherit the lighting scale.
// Transmission retains its radiance estimate because albedo remodulation is
// not valid for a path passing through multiple surfaces.
__global__ void rt_resolve_materials(RtFrame f,const RtGuide* guides,const RtSignals* raw,const RtFiltered* filtered,
    const DVec3* composed,const RtGuide* native_guides,const DVec3* emission,DVec3* output) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i>=f.output_width*f.output_height)return;
    const RtGuide g=native_guides[i];
    if(g.depth<=0 || rt_sharp_optics(g)){output[i]=emission[i];return;}
    const DVec3 geometric=rt_unpack_normal(g.geometric);
    const float px=(float(i%f.output_width)+.5f+f.jitter.x)*f.width/f.output_width-.5f-f.jitter.x;
    const float py=(float(i/f.output_width)+.5f+f.jitter.y)*f.height/f.output_height-.5f-f.jitter.y;
    const int bx=int(floorf(px)),by=int(floorf(py));const float fx=px-bx,fy=py-by;
    const int center=rt_guide_index(f,int(floorf(px+.5f)),int(floorf(py+.5f)));
    const float tolerance=f.settings.depth_threshold*fmaxf(g.depth,1e-3f)+2*rt_gradient(guides,center,f.width,f.height);
    DVec3 diffuse{},other{},fallback{};float weight=0;
    for(int y=0;y<2;++y)for(int x=0;x<2;++x) {
        const int q=rt_guide_index(f,bx+x,by+y);const RtGuide old=guides[q];
        const float w=(x?fx:1-fx)*(y?fy:1-fy);
        fallback=add(fallback,mul(composed[q],w));
        if(g.transparent || old.transparent || old.depth<=0 || g.object_id!=old.object_id ||
            g.material!=old.material || dot(geometric,rt_unpack_normal(old.geometric))<.8f || fabsf(g.depth-old.depth)>tolerance)continue;
        const RtFiltered value=filtered[q];
        diffuse=add(diffuse,mul(f.deterministic_direct?value.c[1]:add(value.c[0],value.c[1]),w));
        other=add(other,mul(add(value.c[2],value.c[3]),w));weight+=w;
        if(f.deterministic_direct) {
            diffuse=add(diffuse,mul(rt_demodulate(raw[q].c[0],old,0),w));
            other=add(other,mul(rt_safe(sub(raw[q].direct,raw[q].c[0])),w));
        }
    }
    output[i]=weight>1e-5f?rt_safe(add(emission[i],divv(add(product(diffuse,rt_albedo(g)),other),weight))):fallback;
}

__global__ void rt_reconstruct(RtFrame f,const RtGuide* guides,const DVec3* composed,
    const RealtimeDiagnosticPixel* diagnostics,const DVec3* previous_color,const RtOutputGuide* previous_guide,
    DVec3* color,RtOutputGuide* output_guide,int diagnostic_width,int diagnostic_height) {
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
        current=add(current,mul(composed[q],w)); sumw+=w;
    }
    current=sumw>1e-5f?divv(current,sumw):composed[center];
    DVec3 history_color{}; float history_weight=0,history_length=0;
    const bool upscale=f.settings.internal_scale<1;
    const bool temporal_enabled=upscale?f.settings.temporal_upscale:f.settings.taa;
    if(f.valid_history && temporal_enabled && f.stationary) {
        // With an unchanged camera and scene, this output pixel covers the
        // same footprint even when jitter selects another side of an edge.
        // Testing the single sampled surface here discards antialiasing history.
        history_color=previous_color[i];history_weight=1;history_length=previous_guide[i].history;
    } else if(f.valid_history && temporal_enabled && g.previous_depth>=0) {
        // Final history is unjittered: do not apply the input jitter correction twice.
        const float ox=(u+g.motion.x)*f.output_width-.5f,oy=(v+g.motion.y)*f.output_height-.5f;
        const int x0=int(floorf(ox)),y0=int(floorf(oy)); const float tx=ox-x0,ty=oy-y0;
        const float tolerance=f.settings.depth_threshold*fmaxf(g.previous_depth,1e-3f)+2*rt_gradient(guides,center,f.width,f.height);
        for(int y=0;y<2;++y) for(int x=0;x<2;++x) {
            const int qx=x0+x,qy=y0+y;
            if(qx<0 || qy<0 || qx>=f.output_width || qy>=f.output_height) continue;
            const int q=qy*f.output_width+qx; const RtOutputGuide old=previous_guide[q];
            bool valid=(g.depth>0)==(old.depth>0);
            // Coverage history follows geometry. A normal map sampled at a
            // different subpixel position must not masquerade as disocclusion.
            if(g.depth>0) valid=valid && g.object_id==old.object_id && g.material==old.material &&
                dot(rt_unpack_normal(g.previous_geometric),old.normal)>f.settings.normal_threshold &&
                fabsf(g.previous_depth-old.depth)<=tolerance;
            if(!valid) continue;
            const float w=(x?tx:1-tx)*(y?ty:1-ty); history_color=add(history_color,mul(previous_color[q],w)); history_weight+=w;
            history_length+=old.history*w;
        }
    }
    if(history_weight>1e-5f) {
        history_color=divv(history_color,history_weight);
        if(!f.stationary) {
            DVec3 mean{},second{};
            for(int y=-1;y<=1;++y)for(int x=-1;x<=1;++x) {
                const DVec3 c=composed[rt_guide_index(f,center%f.width+x,center/f.width+y)];
                mean=add(mean,c);second=add(second,product(c,c));
            }
            mean=divv(mean,9);second=divv(second,9);
            const DVec3 variance=sub(second,product(mean,mean));
            DVec3 sigma=v3(sqrtf(fmaxf(0,variance.x)),sqrtf(fmaxf(0,variance.y)),sqrtf(fmaxf(0,variance.z)));
            if(f.settings.denoiser==RealtimeDenoiser::Optix && !rt_sharp_optics(g)) {
                // Neural output has correlated, smooth residual noise. Its 3x3
                // spatial variance can be almost zero even while the whole
                // patch changes, repeatedly clipping valid moving history.
                // Bound the clip by the robust input variance scaled for the
                // output EMA; geometry validation and reactive edits still win.
                const int qx=min(diagnostic_width-1,center%f.width*diagnostic_width/f.width);
                const int qy=min(diagnostic_height-1,center/f.width*diagnostic_height/f.height);
                const float a=f.settings.taa_current_weight;
                const float noise=sqrtf(fmaxf(0,diagnostics[qy*diagnostic_width+qx].variance)*a/(2-a));
                sigma=v3(fmaxf(sigma.x,noise),fmaxf(sigma.y,noise),fmaxf(sigma.z,noise));
            }
            const float k=f.settings.taa_clip_sigma;
            const bool sharp=g.transparent || g.roughness<.06f;
            history_color=v3(
                fminf(fmaxf(history_color.x,sharp?0:mean.x-k*sigma.x),mean.x+k*sigma.x),
                fminf(fmaxf(history_color.y,sharp?0:mean.y-k*sigma.y),mean.y+k*sigma.y),
                fminf(fmaxf(history_color.z,sharp?0:mean.z-k*sigma.z),mean.z+k*sigma.z));
        }
        const float dx=(float(center%f.width)+.5f)*diagnostic_width/f.width-.5f;
        const float dy=(float(center/f.width)+.5f)*diagnostic_height/f.height-.5f;
        float reactive=0;
        // Every contributing lighting sample can invalidate the reconstructed
        // pixel. Nearest-only masks left stale shadows between input pixels.
        const int taps=diagnostic_width==f.width && diagnostic_height==f.height?1:2;
        for(int y=0;y<taps;++y)for(int x=0;x<taps;++x) {
            const int qx=taps==1?center%f.width:min(diagnostic_width-1,max(0,int(floorf(dx))+x));
            const int qy=taps==1?center/f.width:min(diagnostic_height-1,max(0,int(floorf(dy))+y));
            reactive=fmaxf(reactive,diagnostics[qy*diagnostic_width+qx].reactive);
        }
        float current_weight=f.settings.taa_current_weight;
        // Match the mean age of the two original histories with one resampler:
        // (Hspec-1)+(Htaa-1) = Hcombined-1. Reactive changes still replace it.
        if(rt_sharp_optics(g) && !g.transparent && f.settings.denoise && f.settings.temporal &&
            f.settings.denoiser==RealtimeDenoiser::Svgf)
            current_weight=1.0f/(1.0f/current_weight+f.settings.specular_history-1);
        float alpha=fmaxf(current_weight,reactive);
        if(g.transparent) alpha=fmaxf(alpha,f.settings.split_dielectric?.15f:.5f);
        // In a stationary scene all frame differences are sampling noise and
        // changing pixel coverage, not new shading. Do not clip a converged
        // result to one noisy/jittered neighborhood or restart it reactively.
        // Ramp history in after motion so stopping the camera cannot freeze an
        // old view; edits, animation and camera motion retain the reactive path.
        if(f.stationary && f.settings.denoise && f.settings.temporal && current_weight<1) {
            const float limit=fmaxf(1/current_weight,float(f.settings.diffuse_history));
            history_length=fminf(limit,history_length/history_weight+1);
            alpha=fminf(current_weight,1/history_length);
        } else history_length=1;
        current=rt_mix(history_color,current,alpha);
    } else history_length=1;
    color[i]=rt_safe(current);
    output_guide[i]={g.depth>0?rt_unpack_normal(g.geometric):v3(0,0,0),g.depth,g.object_id,history_length,g.material};
}

__global__ void rt_present(RtFrame f,const DVec3* resolved,const RtGuide* guides,const RtSignals* raw,
    const RtFiltered* temporal,const RtFiltered* filtered,const RealtimeDiagnosticPixel* diagnostics,
    DVec3* output,cudaSurfaceObject_t surface,const DVec3* neural=nullptr,int neural_width=0,int neural_height=0) {
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
    else if(view==RealtimeDebugView::Temporal) c=rt_composite(f.deterministic_direct,raw[p],temporal[p],g);
    else if(view==RealtimeDebugView::Filtered) c=rt_composite(f.deterministic_direct,raw[p],filtered[p],g);
    if(neural && view==RealtimeDebugView::Filtered) {
        const int nx=min(neural_width-1,int((float(x)+.5f)*neural_width/f.output_width));
        const int ny=min(neural_height-1,int((float(y)+.5f)*neural_height/f.output_height));
        c=neural[ny*neural_width+nx];
    }
    c=rt_safe(c); output[i]=c;
    if(surface) surf2Dwrite(make_float4(c.x,c.y,c.z,1),surface,x*int(sizeof(float4)),y);
}

// With no geometry there is no visibility query to submit. Produce the same
// sky radiance and rotational guides as a raygen miss, then keep the normal
// temporal/presentation pipeline. This also avoids a first OptiX launch with
// no acceleration structure, which fails under driver memcheck instrumentation.
__global__ void rt_empty_scene(RtFrame f,DCompactHit* primary,RtGuide* guides,RtSignals* signals,DVec3* emission=nullptr) {
    const int i=int(blockIdx.x*blockDim.x+threadIdx.x);
    if(i>=f.width*f.height)return;
    const DRay ray=rt_primary(f,i);
    RtGuide g{};float z=0,old_z=0;
    const DVec2 uv=rt_project(f.camera,add(f.camera.eye,ray.direction),z);
    const DVec2 old=rt_project(f.previous_camera,add(f.previous_camera.eye,ray.direction),old_z);
    g.motion={old.x-uv.x,old.y-uv.y};g.previous_depth=old_z>0?0:-1;
    guides[i]=g;
    primary[i]={};primary[i].primitive_kind=-1;
    const DVec3 sky=f.scene.environment_background_visible?environment_radiance(f.scene,ray.direction):v3(0,0,0);
    if(emission)emission[i]=sky;
    if(signals) {
        RtSignals value{};
        for(int sample=0;sample<f.settings.samples_per_pixel;++sample)value.c[4]=add(value.c[4],sky);
        value.c[4]=rt_safe(mul(value.c[4],1.0f/f.settings.samples_per_pixel));
        signals[i]=value;
    }
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

class OptixRealtimeRenderer::Impl {
public:
    explicit Impl(CudaDeviceContext context):context_(std::move(context)) {
        std::string reason;
        if(!optix_realtime_available(context_.device_id(),&reason))
            throw std::runtime_error("OptiX RTRT unavailable: "+reason);
        context_.activate(); statistics_.device_id=context_.device_id();
        check_cuda(cudaStreamCreateWithFlags(&stream_,cudaStreamNonBlocking),"create RTRT stream");
    }
    ~Impl() {
        context_.activate();
        if(stream_) {
            cudaStreamSynchronize(stream_);
            denoiser_.reset();lighting_denoiser_.reset();optix_.reset();scene_.reset();
            cudaStreamDestroy(stream_);
        }
    }
    void reset(const RenderSceneSnapshot& snapshot,const RenderSettings& settings) {
        context_.activate();
        check_cuda(cudaStreamSynchronize(stream_),"reset RTRT stream");
        require_device(settings);
        if(!optix_)optix_=std::make_unique<OptixRealtimeBackend>(context_);
        if(!optix_->sync(snapshot,reinterpret_cast<CudaStreamHandle>(stream_)))
            throw std::runtime_error("OptiX RTRT unavailable: "+optix_->reason());
        scene_=std::make_unique<CudaSceneStorage>(snapshot,stream_,statistics_,CudaSceneStorageMode::ShadingOnly);
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
        if(!optix_) optix_=std::make_unique<OptixRealtimeBackend>(context_);
        if(!optix_->sync(snapshot,reinterpret_cast<CudaStreamHandle>(stream_)))
            throw std::runtime_error("OptiX RTRT unavailable: "+optix_->reason());
        statistics_.realtime.hardware_ray_tracing_active=true;
        statistics_.realtime.hardware_ray_tracing_bytes=optix_->resident_bytes();
        statistics_.realtime.hardware_ray_tracing_detail.clear();
        statistics_.realtime.rt_core_version=optix_->rt_core_version();
        statistics_.realtime.ser_supported=optix_->ser_supported();
        statistics_.realtime.ser_active=s.shader_execution_reordering && optix_->ser_supported();
        statistics_.realtime.gas_builds=optix_->gas_builds();
        statistics_.realtime.ias_builds=optix_->ias_builds();
        statistics_.realtime.ias_updates=optix_->ias_updates();
        statistics_.realtime.acceleration_ms=optix_->acceleration_ms();
        const bool full_materials=s.full_resolution_materials && (w!=settings.width || h!=settings.height);
        const bool neural_requested=s.denoise && s.denoiser==RealtimeDenoiser::Optix;
        const bool filter_surfaces=s.specular_antialiasing && std::any_of(snapshot.instances.begin(),snapshot.instances.end(),
            [](const auto& instance) {return std::any_of(instance.materials.begin(),instance.materials.end(),
                [](const Material& m) {return m.normal_texture_id>=0 || m.bump_texture_id>=0;});});
        const bool previous_neural=statistics_.realtime.optix_denoiser_active;
        bool neural=false;
        const bool shading_changed=has_scene_change(changes,SceneChange::Lighting)||has_scene_change(changes,SceneChange::Materials)||
            has_scene_change(changes,SceneChange::MaterialBindings)||has_scene_change(changes,SceneChange::Textures);
        // Scene notifications include Lighting for every rigid transform (shadow
        // and emitter movement). Preserve reprojectable history for that case.
        const bool neural_shading_changed=has_scene_change(changes,SceneChange::Materials)||
            has_scene_change(changes,SceneChange::MaterialBindings)||has_scene_change(changes,SceneChange::Textures)||
            revisions_.lighting!=snapshot.revisions.lighting ||
            (has_scene_change(changes,SceneChange::Lighting) && !has_scene_change(changes,SceneChange::InstanceTransforms));
        const bool resize=w!=width_ || h!=height_ || settings.width!=output_width_ || settings.height!=output_height_;
        auto history_settings=s; history_settings.debug_view=RealtimeDebugView::Final; history_settings.sharpening=0;
        const bool projection_change=valid_ && (std::abs(previous_camera_.viewport_width-camera.viewport_width())>1e-5f || std::abs(previous_camera_.viewport_height-camera.viewport_height())>1e-5f);
        bool reset_history=!valid_ || different_source || resize || state.camera_cut || state.reset_requested || projection_change ||
            history_settings!=history_settings_ ||
            has_scene_change(changes,SceneChange::Geometry) || has_scene_change(changes,SceneChange::Environment);
        // Reserve the ordinary renderer/fallback buffers first on resize. An
        // optional neural allocation must not starve the SVGF fallback.
        if(denoiser_ && (resize || full_materials!=(material_guides_.size()!=0) || !neural_requested))
            denoiser_->release(reinterpret_cast<CudaStreamHandle>(stream_));
        if(lighting_denoiser_ && (resize || !full_materials || !neural_requested))
            lighting_denoiser_->release(reinterpret_cast<CudaStreamHandle>(stream_));
        if(resize) {
            check_cuda(cudaStreamSynchronize(stream_),"resize RTRT buffers");
            width_=w; height_=h; output_width_=settings.width; output_height_=settings.height;
            const std::size_t n=std::size_t(w)*h,o=std::size_t(output_width_)*output_height_;
            for(int j=0;j<2;++j) { guides_[j].resize_exact(n,statistics_); history_[j].resize_exact(n,statistics_); filter_[j].resize_exact(n,statistics_); taa_[j].resize_exact(o,statistics_); output_guides_[j].resize_exact(o,statistics_); }
            primary_.resize_exact(n,statistics_); raw_.resize_exact(n,statistics_); prepared_.resize_exact(n,statistics_);
            composed_.resize_exact(n,statistics_);
            temporal_.resize_exact(n,statistics_); diagnostics_.resize_exact(n,statistics_); output_.resize_exact(o,statistics_);
            statistics_.realtime.framebuffer_bytes=n*(2*sizeof(RtGuide)+2*sizeof(RtHistory)+3*sizeof(RtFiltered)+sizeof(RtPrepared)+sizeof(DVec3)+sizeof(DCompactHit)+sizeof(RtSignals)+sizeof(RealtimeDiagnosticPixel))+
                o*(3*sizeof(DVec3)+2*sizeof(RtOutputGuide));
        }
        const std::size_t material_count=full_materials?std::size_t(settings.width)*settings.height:0;
        if(material_guides_.size()!=material_count) {
            check_cuda(cudaStreamSynchronize(stream_),"resize RTRT material buffers");
            material_guides_.resize_exact(material_count,statistics_);
            material_primary_.resize_exact(material_count,statistics_);
            optical_pixels_.resize_exact(material_count,statistics_);
            optical_count_.resize_exact(material_count?1:0,statistics_);
            material_emission_.resize_exact(material_count,statistics_);material_composed_.resize_exact(material_count,statistics_);
        }
        const std::size_t neural_guide_count=neural_requested?material_count:0;
        if(material_previous_guides_.size()!=neural_guide_count) {
            check_cuda(cudaStreamSynchronize(stream_),"resize OptiX material history guides");
            material_previous_guides_.resize_exact(neural_guide_count,statistics_);
        }
        const std::size_t surface_count=filter_surfaces?std::max(material_count,std::size_t(w)*h):0;
        if(surface_filter_.size()!=surface_count) {
            check_cuda(cudaStreamSynchronize(stream_),"resize specular antialiasing scratch");
            surface_filter_.resize_exact(surface_count,statistics_);
        }
        statistics_.realtime.framebuffer_bytes=std::size_t(w)*h*(2*sizeof(RtGuide)+2*sizeof(RtHistory)+3*sizeof(RtFiltered)+sizeof(RtPrepared)+sizeof(DVec3)+sizeof(DCompactHit)+sizeof(RtSignals)+sizeof(RealtimeDiagnosticPixel))+
            std::size_t(settings.width)*settings.height*(3*sizeof(DVec3)+2*sizeof(RtOutputGuide))+
            material_count*(sizeof(RtGuide)+2*sizeof(DVec3)+sizeof(DCompactHit)+sizeof(int))+(material_count?sizeof(unsigned):0)+
            neural_guide_count*sizeof(RtOutputGuide)+surface_count*sizeof(RtSurfaceFilter);
        if(neural_requested) {
            if(!denoiser_)denoiser_=std::make_unique<OptixRealtimeDenoiser>(context_);
            const auto generation=denoiser_->allocation_generation();
            neural=denoiser_->prepare(full_materials?settings.width:w,full_materials?settings.height:h,
                s.temporal,reinterpret_cast<CudaStreamHandle>(stream_));
            statistics_.allocation_generation+=denoiser_->allocation_generation()-generation;
            statistics_.realtime.optix_denoiser_detail=neural?"":denoiser_->reason()+"; using SVGF";
            if(neural && full_materials) {
                if(!lighting_denoiser_)lighting_denoiser_=std::make_unique<OptixRealtimeDenoiser>(context_);
                const auto lighting_generation=lighting_denoiser_->allocation_generation();
                neural=lighting_denoiser_->prepare(w,h,s.temporal,reinterpret_cast<CudaStreamHandle>(stream_),true);
                statistics_.allocation_generation+=lighting_denoiser_->allocation_generation()-lighting_generation;
                if(!neural)statistics_.realtime.optix_denoiser_detail=lighting_denoiser_->reason()+"; using SVGF";
            }
        } else statistics_.realtime.optix_denoiser_detail.clear();
        statistics_.realtime.optix_denoiser_bytes=(denoiser_?denoiser_->resident_bytes():0)+
            (lighting_denoiser_?lighting_denoiser_->resident_bytes():0);
        statistics_.realtime.framebuffer_bytes+=statistics_.realtime.optix_denoiser_bytes;
        reset_history=reset_history || neural!=previous_neural || (neural && neural_shading_changed);
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
        const bool jitter_enabled=s.internal_scale<1?s.temporal_upscale:s.taa;
        const DVec2 jitter=jitter_enabled?DVec2{rt_halton(sequence%1024+1,2)-.5f,rt_halton(sequence%1024+1,3)-.5f}:DVec2{0,0};
        RtFrame f{scene_->view(),rt_camera(camera),valid_?previous_camera_:rt_camera(camera),instances_.get(),s,
            width_,height_,output_width_,output_height_,sequence,settings.path.sample_seed_offset,jitter,previous_jitter_,valid_?1:0,
            shading_changed?1:0};
        f.settings.denoiser=neural?RealtimeDenoiser::Optix:RealtimeDenoiser::Svgf;
        f.denoiser_reset=neural_shading_changed?1:0;
        const auto same_vector=[](DVec3 a,const Vec3& b) {return a.x==b.x() && a.y==b.y() && a.z==b.z();};
        f.stationary=valid_ && changes==SceneChange::None && same_vector(previous_camera_.eye,camera.eye()) &&
            same_vector(previous_camera_.forward,camera.forward()) && same_vector(previous_camera_.up,camera.up());
        f.deterministic_direct=!s.soft_shadows && f.scene.emissive_light_count==0 &&
            (f.scene.environment_intensity<=0 || (!f.scene.environment_texels &&
                f.scene.environment.x<=0 && f.scene.environment.y<=0 && f.scene.environment.z<=0));
        const int blocks=(width_*height_+kThreadsPerBlock-1)/kThreadsPerBlock;
        const int output_blocks=(output_width_*output_height_+kThreadsPerBlock-1)/kThreadsPerBlock;
        const dim3 prepare_threads(kPrepareWidth,kPrepareHeight);
        const dim3 prepare_blocks((width_+kPrepareWidth-1)/kPrepareWidth,(height_+kPrepareHeight-1)/kPrepareHeight);
        f.hardware_scene=optix_->traversable();
        f.settings.shader_execution_reordering=statistics_.realtime.ser_active;
        const RtOptixParameters optix_parameters{f,primary_.get(),guides_[write].get(),raw_.get()};
        RtFrame material_frame=f;material_frame.width=output_width_;material_frame.height=output_height_;
        // Reuse pending timing events only after they complete; presentation never
        // synchronizes merely to collect statistics.
        const bool timing=!timers_[5].pending();
        if(timing) { timers_[5].begin(stream_); timers_[0].begin(stream_); }
        if(f.hardware_scene) {
            optix_->launch(OptixRealtimePass::Primary,&optix_parameters,sizeof(optix_parameters),width_,height_,reinterpret_cast<CudaStreamHandle>(stream_));
            if(full_materials) {
                const RtOptixParameters p{material_frame,material_primary_.get(),material_guides_.get(),nullptr,material_emission_.get()};
                optix_->launch(OptixRealtimePass::Primary,&p,sizeof(p),output_width_,output_height_,reinterpret_cast<CudaStreamHandle>(stream_));
            }
        } else {
            rt_empty_scene<<<blocks,kThreadsPerBlock,0,stream_>>>(f,primary_.get(),guides_[write].get(),raw_.get());
            if(full_materials)rt_empty_scene<<<output_blocks,kThreadsPerBlock,0,stream_>>>(material_frame,material_primary_.get(),material_guides_.get(),nullptr,material_emission_.get());
        }
        if(filter_surfaces && f.hardware_scene) {
            rt_filter_specular_surface<<<prepare_blocks,prepare_threads,0,stream_>>>(f,guides_[write].get(),surface_filter_.get());
            rt_apply_specular_surface<<<blocks,kThreadsPerBlock,0,stream_>>>(f,primary_.get(),guides_[write].get(),surface_filter_.get());
            if(full_materials) {
                const dim3 material_blocks((output_width_+kPrepareWidth-1)/kPrepareWidth,(output_height_+kPrepareHeight-1)/kPrepareHeight);
                rt_filter_specular_surface<<<material_blocks,prepare_threads,0,stream_>>>(material_frame,material_guides_.get(),surface_filter_.get());
                rt_apply_specular_surface<<<output_blocks,kThreadsPerBlock,0,stream_>>>(material_frame,material_primary_.get(),material_guides_.get(),surface_filter_.get());
            }
        }
        if(timing) { timers_[0].end(stream_); timers_[1].begin(stream_); }
        if(f.hardware_scene)optix_->launch(OptixRealtimePass::Lighting,&optix_parameters,sizeof(optix_parameters),width_,height_,reinterpret_cast<CudaStreamHandle>(stream_));
        if(full_materials && f.hardware_scene) {
            check_cuda(cudaMemsetAsync(optical_count_.get(),0,sizeof(unsigned),stream_),"clear native optical count");
            rt_compact_optics<<<output_blocks,kThreadsPerBlock,0,stream_>>>(material_guides_.get(),output_width_*output_height_,optical_pixels_.get(),optical_count_.get());
            const int optical_threads=std::min(65536,output_width_*output_height_);
            const RtOptixParameters p{material_frame,material_primary_.get(),material_guides_.get(),nullptr,material_emission_.get(),optical_pixels_.get(),optical_count_.get()};
            optix_->launch(OptixRealtimePass::NativeOptics,&p,sizeof(p),optical_threads,1,reinterpret_cast<CudaStreamHandle>(stream_));
        }
        if(timing) { timers_[1].end(stream_); timers_[2].begin(stream_); }
        const auto svgf_temporal=[&] {
            rt_prepare_signal<<<prepare_blocks,prepare_threads,0,stream_>>>(f,guides_[write].get(),raw_.get(),prepared_.get(),guides_[index_].get(),history_[index_].get());
            rt_temporal<<<blocks,kThreadsPerBlock,0,stream_>>>(f,guides_[write].get(),guides_[index_].get(),prepared_.get(),history_[index_].get(),history_[write].get(),temporal_.get(),diagnostics_.get());
        };
        if(neural) {
            // A single unbounded path can move an entire neural reconstruction
            // patch. Apply the same robust input filter before either denoiser.
            rt_prepare_signal<<<prepare_blocks,prepare_threads,0,stream_>>>(f,guides_[write].get(),raw_.get(),prepared_.get(),guides_[index_].get(),history_[index_].get());
            rt_optix_signals<<<blocks,kThreadsPerBlock,0,stream_>>>(f,guides_[write].get(),raw_.get(),prepared_.get(),temporal_.get(),diagnostics_.get());
        }
        else
            svgf_temporal();
        if(timing) { timers_[2].end(stream_); timers_[3].begin(stream_); }
        const RtFiltered* input=temporal_.get();
        const DVec3* neural_output=nullptr;
        if(neural) {
            rt_compose<<<blocks,kThreadsPerBlock,0,stream_>>>(f,guides_[write].get(),raw_.get(),input,composed_.get());
            std::string failure;
            if(full_materials) {
                // Denoise independent samples before upsampling; correlated
                // bilinear noise is a poor input to the neural model. AOVs keep
                // illumination separate so native textures can be remodulated.
                rt_optix_pack_aovs<<<blocks,kThreadsPerBlock,0,stream_>>>(width_*height_,guides_[write].get(),input,
                    static_cast<DVec3*>(lighting_denoiser_->aov_input()));
                rt_optix_guides<<<blocks,kThreadsPerBlock,0,stream_>>>(f,guides_[write].get(),guides_[index_].get(),lighting_denoiser_->guides());
                neural=lighting_denoiser_->invoke(composed_.get(),f.valid_history!=0,reinterpret_cast<CudaStreamHandle>(stream_));
                if(neural) {
                    rt_optix_unpack_aovs<<<blocks,kThreadsPerBlock,0,stream_>>>(width_*height_,guides_[write].get(),
                        static_cast<const DVec3*>(lighting_denoiser_->aov_output()),input,filter_[0].get());
                    input=filter_[0].get();
                    rt_compose<<<blocks,kThreadsPerBlock,0,stream_>>>(f,guides_[write].get(),raw_.get(),input,composed_.get());
                    rt_resolve_materials<<<output_blocks,kThreadsPerBlock,0,stream_>>>(f,guides_[write].get(),raw_.get(),input,composed_.get(),material_guides_.get(),material_emission_.get(),material_composed_.get());
                } else failure=lighting_denoiser_->reason();
            }
            const auto& nf=full_materials?material_frame:f;
            const auto* ng=full_materials?material_guides_.get():guides_[write].get();
            const auto* noisy=full_materials?material_composed_.get():composed_.get();
            const int nb=full_materials?output_blocks:blocks;
            if(neural) {
                if(full_materials)
                    rt_optix_guides<<<nb,kThreadsPerBlock,0,stream_>>>(nf,ng,material_previous_guides_.get(),denoiser_->guides());
                else rt_optix_guides<<<nb,kThreadsPerBlock,0,stream_>>>(nf,ng,guides_[index_].get(),denoiser_->guides());
                neural=denoiser_->invoke(noisy,f.valid_history!=0,reinterpret_cast<CudaStreamHandle>(stream_));
                if(!neural)failure=denoiser_->reason();
            }
            if(neural) {
                auto* finished=full_materials?material_composed_.get():composed_.get();
                rt_optix_finish<<<nb,kThreadsPerBlock,0,stream_>>>(nf.width*nf.height,ng,noisy,
                    static_cast<const DVec3*>(denoiser_->output()),finished,full_materials,full_materials?material_previous_guides_.get():nullptr);
                neural_output=finished;
            } else {
                // Rebuild this frame with SVGF on failure, without mixing either
                // denoiser's history into the fallback or output TAA.
                statistics_.realtime.optix_denoiser_detail=failure+"; using SVGF";
                statistics_.realtime.framebuffer_bytes-=statistics_.realtime.optix_denoiser_bytes;
                statistics_.realtime.optix_denoiser_bytes=denoiser_->resident_bytes()+(lighting_denoiser_?lighting_denoiser_->resident_bytes():0);
                statistics_.realtime.framebuffer_bytes+=statistics_.realtime.optix_denoiser_bytes;
                ++statistics_.realtime.history_resets;
                f.valid_history=0;material_frame.valid_history=0;
                f.settings.denoiser=material_frame.settings.denoiser=RealtimeDenoiser::Svgf;
                input=temporal_.get();
                svgf_temporal();
            }
        }
        const int iterations=s.denoise && !neural?std::max(s.diffuse_iterations,s.specular_iterations):0;
        for(int pass=0;pass<iterations;++pass) {
            RtFiltered* out=filter_[pass%2].get();
            if(pass==0)rt_atrous<0><<<prepare_blocks,prepare_threads,0,stream_>>>(f,guides_[write].get(),input,out,history_[write].get(),pass);
            else if(pass==1)rt_atrous<1><<<prepare_blocks,prepare_threads,0,stream_>>>(f,guides_[write].get(),input,out,history_[write].get(),pass);
            else rt_atrous<><<<blocks,kThreadsPerBlock,0,stream_>>>(f,guides_[write].get(),input,out,history_[write].get(),pass);
            input=out;
        }
        if(timing) { timers_[3].end(stream_); timers_[4].begin(stream_); }
        if(!neural) {
            rt_compose<<<blocks,kThreadsPerBlock,0,stream_>>>(f,guides_[write].get(),raw_.get(),input,composed_.get());
            if(full_materials)rt_resolve_materials<<<output_blocks,kThreadsPerBlock,0,stream_>>>(f,guides_[write].get(),raw_.get(),input,composed_.get(),material_guides_.get(),material_emission_.get(),material_composed_.get());
        }
        rt_reconstruct<<<output_blocks,kThreadsPerBlock,0,stream_>>>(full_materials?material_frame:f,
            full_materials?material_guides_.get():guides_[write].get(),neural_output?neural_output:(full_materials?material_composed_.get():composed_.get()),
            diagnostics_.get(),taa_[index_].get(),output_guides_[index_].get(),taa_[write].get(),output_guides_[write].get(),width_,height_);
        rt_present<<<output_blocks,kThreadsPerBlock,0,stream_>>>(f,taa_[write].get(),guides_[write].get(),raw_.get(),temporal_.get(),input,diagnostics_.get(),output_.get(),cudaSurfaceObject_t(surface),
            neural_output,full_materials?output_width_:width_,full_materials?output_height_:height_);
        if(timing) { timers_[4].end(stream_); timers_[5].end(stream_); }
        check_cuda(cudaGetLastError(),"RTRT frame kernels");
        index_=write; valid_=true; previous_camera_=f.camera; previous_jitter_=jitter;
        source_=snapshot.source_id; revisions_=snapshot.revisions; history_settings_=history_settings;
        previous_instances_.clear();
        for(const auto& instance:snapshot.instances) previous_instances_.emplace(instance.object_id,std::make_pair(instance.object_to_world,instance.normal_to_world));
        statistics_.realtime.active=true; ++statistics_.realtime.frames;
        statistics_.realtime.optix_denoiser_active=neural;
        statistics_.realtime.optix_denoiser_temporal=neural && s.temporal;
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
    std::unique_ptr<OptixRealtimeBackend> optix_;
    std::unique_ptr<OptixRealtimeDenoiser> denoiser_;
    std::unique_ptr<OptixRealtimeDenoiser> lighting_denoiser_;
    std::array<DeviceBuffer<RtGuide>,2> guides_;
    DeviceBuffer<RtGuide> material_guides_;
    DeviceBuffer<RtSurfaceFilter> surface_filter_;
    DeviceBuffer<RtOutputGuide> material_previous_guides_;
    DeviceBuffer<DCompactHit> material_primary_;
    DeviceBuffer<int> optical_pixels_;
    DeviceBuffer<unsigned> optical_count_;
    DeviceBuffer<DVec3> material_emission_,material_composed_;
    std::array<DeviceBuffer<RtHistory>,2> history_;
    std::array<DeviceBuffer<RtFiltered>,2> filter_;
    std::array<DeviceBuffer<DVec3>,2> taa_;
    std::array<DeviceBuffer<RtOutputGuide>,2> output_guides_;
    DeviceBuffer<DCompactHit> primary_;
    DeviceBuffer<RtSignals> raw_;
    DeviceBuffer<RtPrepared> prepared_;
    DeviceBuffer<RtFiltered> temporal_;
    DeviceBuffer<RealtimeDiagnosticPixel> diagnostics_;
    DeviceBuffer<DVec3> output_,composed_;
    DeviceBuffer<RtInstance> instances_;
    PinnedHostBuffer<DVec3> staging_;
    std::array<CudaEventTimer,6> timers_;
    std::vector<RtInstance> instance_host_;
    std::unordered_map<std::uint64_t,std::pair<Mat4,Mat3>> previous_instances_;
    std::uint64_t source_=0;
    SceneRevisions revisions_{};
    RealtimeRenderSettings history_settings_{};
    DCamera previous_camera_{};
    DVec2 previous_jitter_{};
    int width_=0,height_=0,output_width_=0,output_height_=0,index_=0;
    bool valid_=false;
};

OptixRealtimeRenderer::OptixRealtimeRenderer(CudaDeviceContext c):impl_(std::make_unique<Impl>(std::move(c))) {}
OptixRealtimeRenderer::~OptixRealtimeRenderer()=default;
void OptixRealtimeRenderer::reset(const RenderSceneSnapshot& s,const RenderSettings& r){impl_->reset(s,r);}
void OptixRealtimeRenderer::render_next_frame_to_surface(const RenderSceneSnapshot& s,const Camera& c,const RenderSettings& r,const InteractiveFrameState& f,CudaSurfaceHandle target){impl_->render(s,c,r,f,target);}
void OptixRealtimeRenderer::render_next_frame(const RenderSceneSnapshot& s,const Camera& c,const RenderSettings& r,const InteractiveFrameState& f,Framebuffer& target){impl_->render(s,c,r,f,0);impl_->download(target);}
void OptixRealtimeRenderer::download_current_frame(Framebuffer& target){impl_->download(target);}
std::vector<RealtimeDiagnosticPixel> OptixRealtimeRenderer::download_diagnostics(){return impl_->diagnostics();}
CudaStreamHandle OptixRealtimeRenderer::stream_handle() const{return reinterpret_cast<CudaStreamHandle>(impl_->stream_);}
const CudaPathStatistics& OptixRealtimeRenderer::statistics() const{return impl_->statistics_;}
void OptixRealtimeRenderer::refresh_statistics(){impl_->refresh();}
void OptixRealtimeRenderer::set_presentation_state(bool a,bool b){impl_->statistics_.interop_active=a;impl_->statistics_.fallback_active=b;}

} // namespace renderer
