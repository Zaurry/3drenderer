#include "material_resolve.hlsli"
bool MatchesHistorySurface(int2 pixel,SurfaceData surface) {
    if(any(pixel<0) || any(pixel>=int2(g.size.xy)))return false;
    SurfaceData old=ReadSurface(pixel,true);
    if(!surface.identity.w)return old.identity.w==0;
    if(!old.identity.w || any(old.identity.xz!=surface.identity.xz) || asuint(old.virtualNormal.w)!=asuint(surface.virtualNormal.w))return false;
    // Shading normals vary with subpixel normal-map samples. TAA must integrate
    // that variation; use geometric normals for ordinary surface validation.
    float3 a=surface.viewPath.w>0?surface.virtualNormal.xyz:surface.geometricMetallic.xyz;
    float3 b=old.viewPath.w>0?old.virtualNormal.xyz:old.geometricMetallic.xyz;
    return dot(a,b)>.9 && abs(old.positionDepth.w-surface.motion.z)<max(.01,.02*surface.motion.z);
}
[numthreads(8,8,1)]
void Resolve(uint3 id:SV_DispatchThreadID) {
    if(any(id.xy>=g.size.zw))return;
    float2 source=(float2(id.xy)+.5)*float2(g.size.xy)/g.size.zw-.5-(g.options.w?g.jitter.xy:0);
    int2 nearest=clamp(int2(floor(source+.5)),0,int2(g.size.xy)-1);
    SurfaceData surface=ReadSurface(nearest,false);float3 current=ReconstructedColor(nearest);
    SurfaceData motionSurface=surface;
    if(g.options.w!=0) {
        int2 base=int2(floor(source));float2 blend=frac(source);float3 sum=0;float weights=0;
        [unroll]for(int y=0;y<2;++y)[unroll]for(int x=0;x<2;++x) {
            int2 p=clamp(base+int2(x,y),0,int2(g.size.xy)-1);SurfaceData neighbor=ReadSurface(p,false);
            float weight=(x?blend.x:1-blend.x)*(y?blend.y:1-blend.y);
            // Coverage crosses silhouettes. Rejecting background neighbors here
            // turns a jittered low-resolution edge into an aliased staircase.
            if(weight>.05 && neighbor.identity.w && (!motionSurface.identity.w || neighbor.positionDepth.w<motionSurface.positionDepth.w))motionSurface=neighbor;
            sum+=ReconstructedColor(p)*weight;weights+=weight;
        }
        if(weights>1e-6)current=sum/weights;
    }
    float3 detail;SurfaceData detailSurface;
    if(ResolveMaterialDetail(id.xy,detail,detailSurface)){current=detail;motionSurface=detailSurface;}
    Texture2D<float4> history=ResourceDescriptorHeap[g.history.z];RWTexture2D<float4> output=ResourceDescriptorHeap[g.signals.w];
    float3 color=current;float historyLength=1;bool rejected=true;
    if(g.frame.w==0 && (g.options.z==0 || g.options.z==11 || g.options.z==14)) {
        if(g.options.w==0 && g.jitter.z>0){color=lerp(history[id.xy].xyz,current,1/(g.jitter.z+1));historyLength=g.jitter.z+1;rejected=false;}
        else if(g.options.w!=0 && g.pathSampling.z>0) {
            // Jitter changes the sampled surface and normal map within the same
            // fixed output footprint. Do not reject this coverage history or
            // clip a converged color to a single noisy frame's neighborhood.
            // The CPU resets this age on camera motion, scene edits and cuts.
            float4 previous=history[id.xy];
            historyLength=min(min(previous.w+1,max(32,g.sampling.z)),g.pathSampling.z+1);
            color=lerp(previous.xyz,current,1/historyLength);rejected=false;
        }
        else if(g.options.w!=0) {
            float2 previousPixel;
            if(motionSurface.identity.w)previousPixel=float2(id.xy)+motionSurface.motion.xy*float2(g.size.zw)/g.size.xy;
            else {
                // Infinite-distance background tracks camera rotation only.
                float2 uv=(float2(id.xy)+.5)/g.size.zw;
                float3 direction=normalize(g.forward.xyz+g.right.xyz*((uv.x-.5)*g.right.w)+g.up.xyz*((.5-uv.y)*g.up.w));
                previousPixel=Project(g.previousEye.xyz+direction*100000,true)*float2(g.size.zw)/g.size.xy-.5;
            }
            int2 previousSurface=int2(floor((previousPixel+.5)*float2(g.size.xy)/g.size.zw-g.previousJitter.xy));
            bool inside=all(previousPixel>=0)&&all(previousPixel<float2(g.size.zw))&&all(previousSurface>=0)&&all(previousSurface<int2(g.size.xy));
            if(inside) {
                bool valid=MatchesHistorySurface(previousSurface,motionSurface);
                if(!valid && motionSurface.identity.w) {
                    // Match the dilated foreground guide in the previous jitter
                    // footprint. Identity/depth tests still reject disocclusion.
                    [unroll]for(int y=-1;y<=1;++y)[unroll]for(int x=-1;x<=1;++x)
                        if(!valid)valid=MatchesHistorySurface(previousSurface+int2(x,y),motionSurface);
                }
                if(valid) {
                    SamplerState clampSampler=SamplerDescriptorHeap[8];float4 previous=history.SampleLevel(clampSampler,(previousPixel+.5)/g.size.zw,0);
                    float3 minimum=ToYCoCg(current),maximum=minimum;
                    [unroll]for(int y=-1;y<=1;++y)[unroll]for(int x=-1;x<=1;++x){float3 c=ToYCoCg(ReconstructedColor(nearest+int2(x,y)));minimum=min(minimum,c);maximum=max(maximum,c);}
                    float3 extent=max(.001,(maximum-minimum)*.1);float3 clipped=FromYCoCg(clamp(ToYCoCg(previous.xyz),minimum-extent,maximum+extent));
                    float limit=16;
                    if(motionSurface.identity.w) {
                        Material material=LoadMaterial(motionSurface.identity.z);
                        // Emissive surfaces can have zero roughness but no view-
                        // dependent reflection. Preserve their coverage history.
                        bool glossy=(material.pbr.w==1 || material.pbr.w==4) && motionSurface.normalRoughness.w<.08;
                        limit=material.pbr.w==2?2:glossy?4:16;
                    }
                    historyLength=min(previous.w+1,limit);color=lerp(clipped,current,1/historyLength);rejected=false;
                }
            }
        }
    }
    if(g.options.z==1){RWTexture2D<float4> raw=ResourceDescriptorHeap[g.guides.z];color=raw[nearest].xyz;}
    if(g.options.z==2){RWTexture2D<float4> direct=ResourceDescriptorHeap[g.guides.w];color=direct[nearest].xyz;}
    if(g.options.z==3){RWTexture2D<float4> indirect=ResourceDescriptorHeap[g.signals.x];color=indirect[nearest].xyz;}
    if(g.options.z==4)color=surface.albedoOpacity.xyz;
    if(g.options.z==5)color=surface.normalRoughness.xyz*.5+.5;
    if(g.options.z==6)color=surface.positionDepth.www/(1+surface.positionDepth.www);
    if(g.options.z==7)color=float3(.5+surface.motion.xy/32,.5);
    if(g.options.z==8)color=float3(.5+surface.specularMotion.xy/32,surface.viewPath.w>0?1:.5);
    if(g.options.x && g.options.z>=9 && g.options.z<=11){RWTexture2D<float4> debug=ResourceDescriptorHeap[g.reservoirs.w];float4 r=debug[nearest];if(g.options.z==9)color=float3(r.y/g.sampling.z,1-r.y/g.sampling.z,0);if(g.options.z==10)color=(log2(1+r.z)/16).xxx;if(g.options.z==11)color=r.w?float3(1,.1*r.w,0):float3(0,.6,.1);}
    if(!g.options.x && g.options.z==11)color=rejected?float3(1,.1,.05):float3(.05,.6,.15);
    if(g.options.z==12){Texture2D<float4> validation=ResourceDescriptorHeap[g.reconstruction.w];color=validation[nearest].xyz;}
    if(g.options.z==13){RWTexture2D<float4> d=ResourceDescriptorHeap[g.signals.y],s=ResourceDescriptorHeap[g.signals.z];color=float3(log2(1+d[nearest].w)/16,log2(1+s[nearest].w)/16,0);}
    if(g.options.z==14)color=float3(historyLength/max(32,g.sampling.z),rejected?1:0,0);
    output[id.xy]=float4(max(0,color),historyLength);
}
