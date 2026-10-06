#ifndef RENDERER_RTXDI_BRIDGE
#define RENDERER_RTXDI_BRIDGE
#include "lighting.hlsli"
#include "Rtxdi/Utils/RandomSamplerState.hlsli"
#include "Rtxdi/DI/ReSTIRDIParameters.h"
#define RAB_Surface Surface
#define RAB_Material Surface
#define RAB_LightSample LightSample
struct RAB_LightInfo {Light light;uint index;};
// The anchor prevents temporal reuse across an object/material boundary. A rigidly
// moving object is rejected locally; static neighbors can retain their history.
static Surface rabAnchor=(Surface)0;
static uint rabRejection=0;
RAB_Surface RAB_EmptySurface(){return EmptySurface();}
bool RAB_IsSurfaceValid(RAB_Surface s){return s.valid;}
float3 RAB_GetSurfaceWorldPos(RAB_Surface s){return s.position;}
void RAB_SetSurfaceWorldPos(inout RAB_Surface s,float3 p){s.position=p;}
float3 RAB_GetSurfaceNormal(RAB_Surface s){return s.normal;}
void RAB_SetSurfaceNormal(inout RAB_Surface s,float3 n){s.normal=n;}
float3 RAB_GetSurfaceGeoNormal(RAB_Surface s){return s.geometric;}
float3 RAB_GetSurfaceViewDir(RAB_Surface s){return s.view;}
float RAB_GetSurfaceLinearDepth(RAB_Surface s){return s.depth;}
float RAB_GetSurfaceRoughness(RAB_Surface s){return s.type==2 || (s.type==1 && LoadMaterial(s.material).emissionRoughness.w<=0)?0:s.roughness;}
RAB_Material RAB_GetMaterial(RAB_Surface s){return s;}
bool RAB_AreMaterialsSimilar(RAB_Material a,RAB_Material b){return a.type==b.type && abs(a.roughness-b.roughness)<.2 && length(a.f0-b.f0)<.3;}
RAB_Surface RAB_GetGBufferSurface(int2 pixel,bool previous) {
    if(any(pixel<0)||any(pixel>=int2(g.size.xy)))return EmptySurface();
    Surface s=UnpackSurface(ReadSurface(pixel,previous),previous);
    if(previous && rabAnchor.valid) {
        if(s.instance!=rabAnchor.instance || s.material!=rabAnchor.material || s.psrChain!=rabAnchor.psrChain){rabRejection=2;return EmptySurface();}
        Instance instance=LoadInstance(rabAnchor.instance);
        float3 oldPosition=TransformPoint(instance.previous,TransformPoint(instance.inverse,rabAnchor.position));
        if(distance(oldPosition,rabAnchor.position)>1e-4){rabRejection=3;return EmptySurface();}
    }
    return s;
}
int2 RAB_ClampSamplePositionIntoView(int2 p,bool previous){return clamp(p,0,int2(g.size.xy)-1);}
RAB_LightInfo RAB_EmptyLightInfo(){return (RAB_LightInfo)0;}
RAB_LightSample RAB_EmptyLightSample(){return (LightSample)0;}
RAB_LightInfo RAB_LoadLightInfo(uint index,bool previous) {
    StructuredBuffer<Light> lights=ResourceDescriptorHeap[previous?g.lightHistory.x:g.lighting.x];
    RAB_LightInfo info;info.light=lights[index];info.index=index;return info;
}
int RAB_TranslateLightIndex(uint index,bool currentToPrevious) {
    if(!g.lightHistory.w)return int(index);
    StructuredBuffer<uint> map=ResourceDescriptorHeap[currentToPrevious?g.lightHistory.z:g.lightHistory.y];return int(map[index]);
}
RAB_LightSample RAB_SamplePolymorphicLight(RAB_LightInfo info,RAB_Surface s,float2 uv){return SampleLightInfo(info.light,info.index,s,uv);}
float3 RAB_GetReflectedLight(RAB_LightSample light,RAB_Surface s) {
    if(light.pdf<=0 || dot(s.geometric,light.direction)<=0)return 0;
    BsdfEval bsdf=EvaluateBsdf(s,light.direction);
    return light.radiance*(bsdf.diffuse+bsdf.specular)*saturate(dot(s.normal,light.direction))/light.pdf;
}
// Reservoirs use the uniform light-UV measure; the solid-angle PDF belongs in
// the target function, and the discrete light-selection PMF is the proposal.
float DirectReuseMisWeight(RAB_LightSample light,RAB_Surface s,float brdfPdf) {
    // DI estimates the light-sampled part of direct illumination. The existing
    // first BRDF continuation (ordinary or PT) supplies its complementary part.
    // Light-only RIS misses narrow glossy reflections at practical budgets.
    return light.delta || g.frame.z<=s.psrDepth+1?1:Mis(light.pdf*SelectionPdf(light.index),brdfPdf);
}
float RAB_GetLightSampleTargetPdfForSurface(RAB_LightSample light,RAB_Surface s){
    return Luminance(RAB_GetReflectedLight(light,s))*DirectReuseMisWeight(light,s,EvaluateBsdf(s,light.direction).pdf);
}
bool RAB_GetConservativeVisibility(RAB_Surface s,RAB_LightSample l){uint rng=Hash(asuint(s.position.x)^asuint(s.position.z)^g.frame.x);return VisibleLight(s,l,rng);}
bool RAB_GetTemporalConservativeVisibility(RAB_Surface current,RAB_Surface old,RAB_LightSample l){return RAB_GetConservativeVisibility(old,l);}
float3 RAB_SurfaceEvaluateBrdfTimesNoL(RAB_Surface s,float3 direction){BsdfEval bsdf=EvaluateBsdf(s,direction);return (bsdf.diffuse+bsdf.specular)*saturate(dot(s.normal,direction));}
// Footprint reconnection must never classify a delta BSDF's zero continuous
// PDF as broad scattering. The SDK requires a large density for this case.
float RAB_SurfaceEvaluateBrdfPdf(RAB_Surface s,float3 direction){return IsDeltaSurface(s)?1e20:EvaluateBsdf(s,direction).pdf;}
RTXDI_RuntimeParameters RuntimeParameters(){RTXDI_RuntimeParameters p=(RTXDI_RuntimeParameters)0;p.neighborOffsetMask=255;p.frameIndex=g.frame.x;return p;}
RTXDI_ReservoirBufferParameters ReservoirParameters(){RTXDI_ReservoirBufferParameters p=(RTXDI_ReservoirBufferParameters)0;p.reservoirBlockRowPitch=g.reservoirs.x;p.reservoirArrayPitch=g.reservoirs.y;return p;}
RWStructuredBuffer<RTXDI_PackedDIReservoir> DiBuffer(){return ResourceDescriptorHeap[g.history.x];}
StructuredBuffer<float2> NeighborBuffer(){return ResourceDescriptorHeap[g.reservoirs.z];}
#define RTXDI_LIGHT_RESERVOIR_BUFFER DiBuffer()
#define RTXDI_NEIGHBOR_OFFSETS_BUFFER NeighborBuffer()
#endif
