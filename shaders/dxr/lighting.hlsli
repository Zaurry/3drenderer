#ifndef RENDERER_DXR_LIGHTING
#define RENDERER_DXR_LIGHTING
#include "trace.hlsli"
Light LoadLight(uint index){StructuredBuffer<Light> b=ResourceDescriptorHeap[g.lighting.x];return b[index];}
float SelectionPdf(uint index){StructuredBuffer<float> cdf=ResourceDescriptorHeap[g.lightExtra.y];return cdf[index]-(index?cdf[index-1]:0);}
uint SelectCdf(uint descriptor,uint count,float u){StructuredBuffer<float> cdf=ResourceDescriptorHeap[descriptor];uint lo=0,hi=count-1;while(lo<hi){uint mid=(lo+hi)/2;if(cdf[mid]<u)lo=mid+1;else hi=mid;}return lo;}
float3 RotateEnvironment(float3 d,float angle){float sn,cs;sincos(angle,sn,cs);return float3(cs*d.x+sn*d.z,d.y,-sn*d.x+cs*d.z);}
struct EnvironmentAliasEntry {float threshold,pmf;uint alias,reserved;};
float2 ReservoirLightUv(float2 uv) {
    // Stay inside the SDK's truncation bin when it packs UVs to 16 bits.
    // Hashed environment samples round this value and its decoded endpoint
    // to the same seed, even when float multiply/divide introduces roundoff.
    return (floor(saturate(uv)*65535)+.25)/65535;
}
float EnvironmentPdf(float3 direction) {
    if(!g.lighting.w)return 1/(4*PI);
    float3 d=RotateEnvironment(direction,-g.environmentInfo.x);float2 uv=float2(atan2(d.z,d.x)/(2*PI)+.5,acos(clamp(d.y,-1,1))/PI);
    uint2 p=min(uint2(uv*g.environmentInfo.zw),uint2(g.environmentInfo.zw)-1);uint index=p.y*(uint)g.environmentInfo.z+p.x;
    StructuredBuffer<EnvironmentAliasEntry> table=ResourceDescriptorHeap[g.lighting.w];float pmf=table[index].pmf;
    float solidAngle=2*PI/g.environmentInfo.z*(cos(PI*p.y/g.environmentInfo.w)-cos(PI*(p.y+1)/g.environmentInfo.w));return pmf/max(solidAngle,1e-20);
}
struct LightSample {float3 direction,radiance,position,shadowPosition;float distance,pdf;bool delta,castsShadows;uint index;float2 uv;};
LightSample SampleLightInfo(Light l,uint index,Surface s,float2 uv) {
    LightSample result=(LightSample)0;result.index=index;result.uv=uv;
    uint type=(uint)l.positionType.w;result.castsShadows=(l.identity.z&2)!=0;
    result.radiance=l.radianceRange.xyz;result.distance=1e30;result.pdf=1;
    if(type==5) {
        float3 d;
        if(g.lighting.w) {
            // RTXDI persists two 16-bit UVs. Expand their combined seed to
            // independent column/coin/jitter draws; splitting one 16-bit value
            // into column and coin would bias HDR maps with >65536 texels.
            uint2 fixedUv=uint2(round(saturate(uv)*65535));uint seed=Hash(fixedUv.x|(fixedUv.y<<16));
            uint count=(uint)(g.environmentInfo.z*g.environmentInfo.w);
            uint column=min(uint(Random(seed)*count),count-1);
            StructuredBuffer<EnvironmentAliasEntry> table=ResourceDescriptorHeap[g.lighting.w];EnvironmentAliasEntry entry=table[column];
            uint texel=Random(seed)<entry.threshold?column:entry.alias;
            uint2 p=uint2(texel%(uint)g.environmentInfo.z,texel/(uint)g.environmentInfo.z);
            float phi=2*PI*((p.x+Random(seed))/g.environmentInfo.z-.5);
            float y=lerp(cos(PI*p.y/g.environmentInfo.w),cos(PI*(p.y+1)/g.environmentInfo.w),Random(seed));
            d=float3(sqrt(max(0,1-y*y))*cos(phi),y,sqrt(max(0,1-y*y))*sin(phi));d=RotateEnvironment(d,g.environmentInfo.x);
        } else {float y=1-2*uv.x,phi=2*PI*uv.y;d=float3(sqrt(max(0,1-y*y))*cos(phi),y,sqrt(max(0,1-y*y))*sin(phi));}
        result.direction=d;result.position=s.position+d*100000;result.shadowPosition=result.position;result.radiance=Environment(d);result.pdf=EnvironmentPdf(d);return result;
    }
    if(type==2) {
        float3 axis=normalize(-l.directionRadius.xyz);float radius=l.directionRadius.w;
        if(radius>0){float3 t,b;Basis(axis,t,b);float c=lerp(1,cos(radius),uv.x),phi=2*PI*uv.y;result.direction=axis*c+(t*cos(phi)+b*sin(phi))*sqrt(max(0,1-c*c));}
        else result.direction=axis;
        // Analytical sun is not intersectable: its integral is sampled only by NEE.
        result.position=s.position+result.direction*100000;result.shadowPosition=result.position;result.delta=true;return result;
    }
    result.position=l.positionType.xyz;
    float3 normal=l.directionRadius.xyz;float area=l.directionRadius.w;
    if(type==3)result.position+=(2*uv.x-1)*l.axisUInner.xyz+(2*uv.y-1)*l.axisVOuter.xyz;
    if(type==4){float u=sqrt(uv.x);float2 bary=float2(u*(1-uv.y),u*uv.y);result.position+=bary.x*l.axisUInner.xyz+bary.y*l.axisVOuter.xyz;
        Surface emitter=EvaluateSurface(asuint(l.axisUInner.w),asuint(l.axisVOuter.w),bary,0,normalize(result.position-s.position));result.radiance=emitter.emission;}
    float3 delta=result.position-s.position;float distance2=dot(delta,delta);result.distance=sqrt(distance2);result.direction=delta/max(result.distance,1e-12);
    result.shadowPosition=result.position;
    if(type<=1) {
        float attenuation=l.radianceRange.w>0?pow(saturate(1-pow(result.distance/l.radianceRange.w,4)),2):1;
        if(type==1){float cosine=dot(-result.direction,l.directionRadius.xyz);float cone=l.axisUInner.w<=l.axisVOuter.w?(cosine>=l.axisVOuter.w?1:0):saturate((cosine-l.axisVOuter.w)/(l.axisUInner.w-l.axisVOuter.w));attenuation*=cone;}
        result.radiance*=attenuation/max(distance2,1e-10);result.delta=true;
        // Match the existing punctual-light contract: radius softens visibility
        // over a receiver-facing disk without changing intensity or range.
        if(l.directionRadius.w>0){float3 t,b;Basis(result.direction,t,b);float radius=l.directionRadius.w*sqrt(uv.x),phi=2*PI*uv.y;result.shadowPosition+=radius*(cos(phi)*t+sin(phi)*b);}
    } else {
        float cosine=dot(normal,-result.direction);if(l.identity.z&1)cosine=abs(cosine);
        if(cosine<=1e-8 || area<=0){result.radiance=0;result.pdf=0;return result;}
        result.pdf=distance2/(cosine*area);
    }
    return result;
}
LightSample SampleLight(uint index,Surface s,float2 uv){return SampleLightInfo(LoadLight(index),index,s,uv);}
bool VisibleLight(Surface s,LightSample light,inout uint rng){float3 delta=light.shadowPosition-s.position;float distance=length(delta);return !light.castsShadows || Visible(s,delta/max(distance,1e-12),distance,rng);}
float HitLightPdf(Surface hit,Surface from,float3 direction) {
    StructuredBuffer<uint> emitters=ResourceDescriptorHeap[g.lightExtra.x];Instance i=LoadInstance(hit.instance);uint index=emitters[i.geometry.w+hit.primitive];
    if(index==0xffffffff)return 0;Light l=LoadLight(index);float cosine=abs(dot(l.directionRadius.xyz,-direction));
    return SelectionPdf(index)*dot(hit.position-from.position,hit.position-from.position)/max(cosine*l.directionRadius.w,1e-20);
}
void DirectLighting(Surface s,inout uint rng,out float3 diffuse,out float3 specular,bool hasBsdfContinuation=true) {
    diffuse=0;specular=0;if(IsDeltaSurface(s) || s.type==3 || g.lighting.y==0)return;
    uint index=SelectCdf(g.lightExtra.y,g.lighting.y,Random(rng));LightSample light=SampleLight(index,s,float2(Random(rng),Random(rng)));
    float pdf=light.pdf*SelectionPdf(index);if(pdf<=0 || MaxComponent(light.radiance)<=0 || dot(s.geometric,light.direction)<=0)return;
    BsdfEval bsdf=EvaluateBsdf(s,light.direction);float factor=saturate(dot(s.normal,light.direction))/pdf;
    if(!light.delta && hasBsdfContinuation)factor*=Mis(pdf,bsdf.pdf);
    if(!VisibleLight(s,light,rng))return;
    diffuse=light.radiance*bsdf.diffuse*factor;specular=light.radiance*bsdf.specular*factor;
}
#endif
