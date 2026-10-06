#undef DXR_INLINE
#define DXR_INLINE 1
#include "rtxdi_bridge.hlsli"
// TemporalResampling does not provide a default for this SDK compile switch.
// Define it before either include so temporal BASIC correction is compiled in.
#define RTXDI_ALLOWED_BIAS_CORRECTION RTXDI_BIAS_CORRECTION_RAY_TRACED
#include "Rtxdi/DI/TemporalResampling.hlsli"
#include "Rtxdi/DI/SpatialResampling.hlsli"

[numthreads(8,8,1)]
void DiInitialTemporal(uint3 id:SV_DispatchThreadID) {
    uint2 pixel=id.xy;if(any(pixel>=g.size.xy))return;
    Surface s=RAB_GetGBufferSurface(pixel,false);rabAnchor=s;
    RTXDI_RandomSamplerState rng=RTXDI_InitRandomSampler(pixel,g.frame.x,1);
    RTXDI_DIReservoir reservoir=RTXDI_EmptyDIReservoir();int2 temporalPixel=-1;
    if(s.valid && !IsDeltaSurface(s) && s.type!=3 && g.lighting.y) {
        for(uint candidate=0;candidate<g.sampling.x;++candidate) {
            uint index=SelectCdf(g.lightExtra.y,g.lighting.y,RTXDI_GetNextRandom(rng));
            float2 uv=float2(RTXDI_GetNextRandom(rng),RTXDI_GetNextRandom(rng));
            uv=ReservoirLightUv(uv);
            LightSample light=SampleLight(index,s,uv);
            RTXDI_StreamSample(reservoir,index,uv,RTXDI_GetNextRandom(rng),RAB_GetLightSampleTargetPdfForSurface(light,s),rcp(max(SelectionPdf(index),1e-20)));
        }
        RTXDI_FinalizeResampling(reservoir,1,reservoir.M);
        // M represents a frame after RIS, independent of the candidate budget.
        reservoir.M=1;
        if(!g.frame.w) {
            RTXDI_DITemporalResamplingParameters p=(RTXDI_DITemporalResamplingParameters)0;
            p.maxHistoryLength=g.sampling.z;p.biasCorrectionMode=RTXDI_BIAS_CORRECTION_BASIC;p.depthThreshold=.02;p.normalThreshold=.9;
            p.enablePermutationSampling=1;p.uniformRandomNumber=Hash(g.frame.x);
            SurfaceData data=ReadSurface(pixel,false);LightSample selected=RAB_EmptyLightSample();
            float2 motion=data.motion.xy+(g.options.w?g.jitter.xy-g.previousJitter.xy:0);
            reservoir=RTXDI_DITemporalResampling(pixel,s,reservoir,rng,RuntimeParameters(),ReservoirParameters(),float3(motion,data.motion.z-data.positionDepth.w),1-(g.frame.x&1),p,temporalPixel,selected);
        }
    }
    RTXDI_StoreDIReservoir(reservoir,ReservoirParameters(),pixel,2);
    RWTexture2D<float4> debug=ResourceDescriptorHeap[g.reservoirs.w];
    debug[pixel]=float4(reservoir.M,reservoir.age,reservoir.weightSum,g.frame.w?1:temporalPixel.x<0?max(2,rabRejection):0);
}
[numthreads(8,8,1)]
void DiSpatialShade(uint3 id:SV_DispatchThreadID) {
    uint2 pixel=id.xy;if(any(pixel>=g.size.xy))return;
    Surface s=RAB_GetGBufferSurface(pixel,false);
    RTXDI_DIReservoir reservoir=RTXDI_LoadDIReservoir(ReservoirParameters(),pixel,2);
    RTXDI_RandomSamplerState rng=RTXDI_InitRandomSampler(pixel,g.frame.x,2);
    if(s.valid && !IsDeltaSurface(s) && s.type!=3 && g.sampling.y) {
        RTXDI_DISpatialResamplingParameters p=(RTXDI_DISpatialResamplingParameters)0;
        p.numSamples=g.sampling.y;p.numDisocclusionBoostSamples=g.sampling.y;p.samplingRadius=16;
        p.biasCorrectionMode=RTXDI_BIAS_CORRECTION_BASIC;p.depthThreshold=.02;p.normalThreshold=.9;p.enableMaterialSimilarityTest=1;p.targetHistoryLength=4;
        LightSample selected=RAB_EmptyLightSample();
        reservoir=RTXDI_DISpatialResampling(pixel,s,reservoir,rng,RuntimeParameters(),ReservoirParameters(),2,p,selected);
    }
    float3 diffuse=0,specular=0;
    if(s.valid && RTXDI_IsValidDIReservoir(reservoir)) {
        LightSample light=SampleLight(RTXDI_GetDIReservoirLightIndex(reservoir),s,RTXDI_GetDIReservoirSampleUV(reservoir));
        // Always trace current visibility. Cached visibility must never freeze a moving shadow.
        bool visible=RAB_GetConservativeVisibility(s,light);uint age=reservoir.age;
        RTXDI_StoreVisibilityInDIReservoir(reservoir,visible?1:0,false);reservoir.age=age;
        if(visible && light.pdf>0 && dot(s.geometric,light.direction)>0) {
            BsdfEval bsdf=EvaluateBsdf(s,light.direction);
            float3 incident=light.radiance*saturate(dot(s.normal,light.direction))*reservoir.weightSum/light.pdf;
            incident*=DirectReuseMisWeight(light,s,bsdf.pdf);
            diffuse=incident*bsdf.diffuse;specular=incident*bsdf.specular;
        }
    }
    RTXDI_StoreDIReservoir(reservoir,ReservoirParameters(),pixel,g.frame.x&1);
    float3 throughput=ReadSurface(pixel,false).throughput.xyz;diffuse*=throughput;specular*=throughput;
    RWTexture2D<float4> raw=ResourceDescriptorHeap[g.guides.z],direct=ResourceDescriptorHeap[g.guides.w];
    RWTexture2D<float4> diffuseOutput=ResourceDescriptorHeap[g.signals.y],specularOutput=ResourceDescriptorHeap[g.signals.z];
    raw[pixel]+=float4(diffuse+specular,0);direct[pixel]+=float4(diffuse+specular,0);
    diffuseOutput[pixel]+=float4(diffuse,0);specularOutput[pixel]+=float4(specular,0);
    RWTexture2D<float4> debug=ResourceDescriptorHeap[g.reservoirs.w];float4 value=debug[pixel];value.xyz=float3(reservoir.M,reservoir.age,reservoir.weightSum);debug[pixel]=value;
}
