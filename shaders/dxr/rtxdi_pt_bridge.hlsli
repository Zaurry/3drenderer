#ifndef RENDERER_RTXDI_PT_BRIDGE
#define RENDERER_RTXDI_PT_BRIDGE
#include "rtxdi_bridge.hlsli"
#include "Rtxdi/PT/ReSTIRPTParameters.h"
#include "Rtxdi/PT/PathTracerRandomContext.hlsli"
#define RAB_RayPayload Hit
#define RAB_DISTANT_LIGHT_DISTANCE 100000.0
float RAB_RayPayloadGetCommittedHitT(Hit h){return h.t;}
RWStructuredBuffer<RTXDI_PackedPTReservoir> PtBuffer(){return ResourceDescriptorHeap[g.history.y];}
#define RTXDI_PT_RESERVOIR_BUFFER PtBuffer()
#include "Rtxdi/PT/Reservoir.hlsli"
struct RAB_PathTracerUserData {uint pathType;float hitDistance;float3 firstDirection;float diffuseGuide,specularGuide;bool guideCaptured;};
void RAB_PathTracerUserDataSetPathType(inout RAB_PathTracerUserData u,uint type){u.pathType=type;}
float3 RAB_LightSamplePosition(LightSample l){return l.position;}
float3 RAB_LightSampleRadiance(LightSample l){return l.radiance;}
float RAB_LightSampleSolidAnglePdf(LightSample l){return l.pdf;}
float3 RAB_GetReflectedBsdfRadianceForSurface(float3 position,float3 radiance,Surface s) {
    float3 direction=SafeNormal(position-s.position,s.normal);
    if(dot(s.geometric,direction)<=0)return 0;
    return radiance*RAB_SurfaceEvaluateBrdfTimesNoL(s,direction);
}
float3 RAB_GetPTSampleTargetPdfForSurface(float3 position,float3 radiance,Surface s){return max(0,RAB_GetReflectedBsdfRadianceForSurface(position,radiance,s));}
bool RAB_GetConservativeVisibility(Surface s,float3 p){float3 v=p-s.position;uint rng=Hash(asuint(p.x)^asuint(p.y)^asuint(p.z));return Visible(s,normalize(v),length(v),rng);}
float RAB_GetMISWeightForNEE(uint index,LightSample light,float3 direction,float lightPdf,float scatterPdf) {
    return light.delta?1:Mis(lightPdf*SelectionPdf(index),scatterPdf);
}
void RAB_LastBounceDenoiserCallback(float3 p,Surface s,inout RAB_PathTracerUserData u){if(u.hitDistance==0)u.hitDistance=distance(p,s.position);}
void RAB_ReconnectionDenoiserCallback(RTXDI_PTReservoir r,Surface s,inout RAB_PathTracerUserData u){RAB_LastBounceDenoiserCallback(r.translatedWorldPosition,s,u);}
uint RAB_GetDuplicationMapCount(int2 p){return 0;} // Uniform decorrelation does not read duplication counts.
#include "Rtxdi/PT/PathTracerContext.hlsli"

RTXDI_BrdfRaySample PtSampleBsdf(Surface s,inout RTXDI_RandomSamplerState rng) {
    RTXDI_BrdfRaySample result=RTXDI_EmptyBrdfRaySample();uint seed=RTXDI_murmur3(rng);float3 weight;bool delta,specularLobe;
    if(SampleBsdf(s,seed,result.outDirection,weight,result.outPdf,delta,specularLobe)) {
        result.brdfTimesNoL=weight*result.outPdf;
        if(delta)result.properties.SetDelta();
        if(specularLobe)result.properties.SetSpecular();
        if(dot(s.geometric,result.outDirection)<0)result.properties.SetTransmission();
    }
    return result;
}
template<typename Context>
void RAB_PathTrace(inout RTXDI_PathTracerContext<Context> ctx,inout RTXDI_PathTracerRandomContext rng,inout RAB_PathTracerUserData user) {
    uint availableDepth=g.frame.z-ctx.GetIntersectionSurface().psrDepth;
    [loop]while(ctx.GetBounceDepth()<=min(availableDepth,ctx.GetMaxPathBounce())) {
        ctx.BeginPathState();Surface previous=ctx.GetIntersectionSurface();
        RTXDI_BrdfRaySample bsdf=PtSampleBsdf(previous,rng.replayRandomSamplerState);
        ctx.SetBrdfRaySample(bsdf);
        if(bsdf.outPdf<=0 || !ctx.ValidContinuationRayBrdfOverPdf())break;
        ctx.MultiplyPathThroughput(ctx.GetContinuationRayBrdfOverPdf());
        if(ctx.GetBounceDepth()>=5 && ctx.ShouldRunRussianRoulette()) {
            float continuation=clamp(MaxComponent(ctx.GetPathThroughput()),.05,.95);
            if(RTXDI_GetNextRandom(rng.initialRandomSamplerState)>continuation)break;
            ctx.MultiplyPathThroughput(rcp(continuation));ctx.RecordRussianRouletteProbability(continuation);
        }
        RayDesc ray;ray.Origin=Offset(previous.position,previous.geometric,bsdf.outDirection);ray.Direction=bsdf.outDirection;ray.TMin=0;ray.TMax=1e30;
        ctx.SetContinuationRay(ray);if(!ctx.AnalyzePathReconnectibilityBeforeTrace())break;
        uint alphaSeed=Hash(rng.replayRandomSamplerState.seed+ctx.GetBounceDepth());
        Hit hit=Trace(ray.Origin,ray.Direction,alphaSeed);ctx.SetTraceResult(hit);
        if(ctx.GetBounceDepth()==2){user.hitDistance=hit.instance==0xffffffff?RAB_DISTANT_LIGHT_DISTANCE:hit.t;user.firstDirection=ray.Direction;}
        // Denoiser guides come from one original BRDF sample, independently of
        // radiance, visibility, and which reservoir wins resampling. The skipped
        // lobe stays at zero and is reconstructed by NRD's prepass.
        if(ctx.GetBounceDepth()==2 && !user.guideCaptured) {
            user.guideCaptured=true;
            if(bsdf.properties.IsSpecular())user.specularGuide=user.hitDistance;
            else user.diffuseGuide=user.hitDistance;
        }
        if(hit.instance==0xffffffff) {
            ctx.RecordPathRadianceMiss(rng.initialRandomSamplerState);
            float weight=bsdf.properties.IsDelta()?1:Mis(bsdf.outPdf,SelectionPdf(g.lightExtra.w)*EnvironmentPdf(ray.Direction));
            ctx.RecordEnvironmentMapLightSample(Environment(ray.Direction)*weight,previous,rng.initialRandomSamplerState);
            break;
        }
        Surface s=EvaluateSurface(hit.instance,hit.primitive,hit.bary,hit.t,ray.Direction);ctx.RecordPathIntersection(s);
        if(ctx.IsPathTerminated())break;
        if(ctx.ShouldSampleEmissiveSurfaces()) {
            float weight=bsdf.properties.IsDelta()?1:Mis(bsdf.outPdf,HitLightPdf(s,previous,ray.Direction));
            ctx.RecordEmissiveLightSample(s.emission*weight,previous,rng.initialRandomSamplerState);
        }
        if(s.type==3)break;
        if(!IsDeltaSurface(s) && g.lighting.y && ctx.ShouldSampleNee()) {
            uint index=SelectCdf(g.lightExtra.y,g.lighting.y,RTXDI_GetNextRandom(rng.initialRandomSamplerState));
            float2 uv=ReservoirLightUv(float2(RTXDI_GetNextRandom(rng.initialRandomSamplerState),RTXDI_GetNextRandom(rng.initialRandomSamplerState)));
            LightSample light=SampleLight(index,s,uv);float selectionPdf=SelectionPdf(index),scatterPdf=RAB_SurfaceEvaluateBrdfPdf(s,light.direction);
            float3 reflected=RAB_GetReflectedLight(light,s);
            if(selectionPdf>0 && any(reflected>0) && RAB_GetConservativeVisibility(s,light)) {
                RTXDI_SampledLightData data;data.lightData=index|0x80000000;data.uvData=uint(uv.x*65535)|(uint(uv.y*65535)<<16);
                float3 radiance=reflected/selectionPdf;
                if(ctx.GetBounceDepth()<availableDepth)radiance*=RAB_GetMISWeightForNEE(index,light,light.direction,light.pdf,scatterPdf);
                ctx.RecordNeeLightSample(data,radiance,selectionPdf,scatterPdf,light,rng.initialRandomSamplerState);
            }
        }
        ctx.IncreaseBounceDepth();
    }
}
RTXDI_PTReconnectionParameters PtReconnectionParameters() {
    RTXDI_PTReconnectionParameters p=(RTXDI_PTReconnectionParameters)0;
    // Use RTXDI 3.1's footprint criterion and default thresholds. Unlike a
    // material-wide roughness cutoff, it recognizes broad diffuse samples on
    // wet PBR surfaces while replaying their narrow specular prefixes.
    p.reconnectionMode=RTXDI_RESTIRPT_RECONNECTION_MODE_FOOTPRINT;
    p.minConnectionFootprint=.02;p.minConnectionFootprintSigma=.2;
    p.minPdfRoughness=.1;p.minPdfRoughnessSigma=.01;return p;
}
RTXDI_PTHybridShiftPerFrameParameters PtShiftParameters(){RTXDI_PTHybridShiftPerFrameParameters p=(RTXDI_PTHybridShiftPerFrameParameters)0;p.maxBounceDepth=g.frame.z+1;p.maxRcVertexLength=g.frame.z+2;return p;}
RTXDI_PTBufferIndices PtIndices() {
    RTXDI_PTBufferIndices b=(RTXDI_PTBufferIndices)0;
    b.initialPathTracerOutputBufferIndex=2;b.initialPathTracerPreservedBufferIndex=2;b.temporalResamplingInputBufferIndex=1-(g.frame.x&1);
    b.temporalResamplingOutputBufferIndex=3;b.spatialResamplingInputBufferIndex=3;b.spatialResamplingOutputBufferIndex=g.frame.x&1;b.finalShadingInputBufferIndex=g.frame.x&1;return b;
}
#endif
