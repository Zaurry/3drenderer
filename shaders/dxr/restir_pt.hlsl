#include "rtxdi_pt_bridge.hlsli"
#include "Rtxdi/PT/InitialSampling.hlsli"
#include "Rtxdi/PT/TemporalResampling.hlsli"
#include "Rtxdi/PT/SpatialResampling.hlsli"
// Uniform decorrelation is independent of a duplication map. These aliases are
// required by the SDK's unselected stagnancy branch; the branch is never run.
RWTexture2D<float> PtUnusedStagnancy(){return ResourceDescriptorHeap[g.reservoirs.w];}
RWTexture2D<float2> PtUnusedDuplication(){return ResourceDescriptorHeap[g.reservoirs.w];}
#define RTXDI_PT_SMOOTHED_DUPLICATION_MAP PtUnusedStagnancy()
#define RTXDI_PT_DUPLICATION_MAP PtUnusedDuplication()
#include "Rtxdi/PT/Decorrelation.hlsli"

void PtInitial(uint2 pixel) {
    Surface s=RAB_GetGBufferSurface(pixel,false);RTXDI_PTReservoir reservoir=RTXDI_EmptyPTReservoir();
    if(s.valid && s.type!=3 && g.frame.z>s.psrDepth+1) {
        RTXDI_PTInitialSamplingParameters p=(RTXDI_PTInitialSamplingParameters)0;p.numInitialSamples=g.frame.y;p.maxBounceDepth=g.frame.z-s.psrDepth+1;p.maxRcVertexLength=g.frame.z-s.psrDepth+2;
        RTXDI_PTInitialSamplingRuntimeParameters runtime;runtime.cameraPos=g.eye.xyz;runtime.prevCameraPos=g.previousEye.xyz;runtime.prevPrevCameraPos=g.previousEye.xyz;
        RTXDI_PathTracerRandomContext rng=RTXDI_InitializePathTracerRandomContext(pixel,g.frame.x,RTXDI_PT_GENERATE_INITIAL_SAMPLES_RANDOM_SEED,RTXDI_PT_GENERATE_INITIAL_SAMPLES_REPLAY_RANDOM_SEED);
        RAB_PathTracerUserData user=(RAB_PathTracerUserData)0;
        reservoir=GenerateInitialSamples(p,runtime,PtReconnectionParameters(),rng,s,user);reservoir.M=1;
        RWTexture2D<float4> diffuseOutput=ResourceDescriptorHeap[g.signals.y],specularOutput=ResourceDescriptorHeap[g.signals.z];
        float4 d=diffuseOutput[pixel],r=specularOutput[pixel];d.w=user.diffuseGuide;r.w=user.specularGuide;
        diffuseOutput[pixel]=d;specularOutput[pixel]=r;
    }
    RTXDI_StorePTReservoir(reservoir,ReservoirParameters(),pixel,2);
}
void PtTemporal(uint2 pixel) {
    Surface s=RAB_GetGBufferSurface(pixel,false);rabAnchor=s;
    RTXDI_PTReservoir reservoir=RTXDI_LoadPTReservoir(ReservoirParameters(),pixel,2);
    if(!g.frame.w && !(g.sampling.w&1) && s.valid && s.type!=3 && g.frame.z>s.psrDepth+1) {
        SurfaceData surface=ReadSurface(pixel,false);
        RTXDI_PTTemporalResamplingParameters p=(RTXDI_PTTemporalResamplingParameters)0;
        p.depthThreshold=.02;p.normalThreshold=.9;p.enablePermutationSampling=1;p.maxHistoryLength=g.sampling.z;p.maxReservoirAge=min(g.sampling.z,30);
        p.enableVisibilityBeforeCombine=1;p.enableAgeBasedRejection=1;p.uniformRandomNumber=Hash(g.frame.x);
        RTXDI_PTTemporalResamplingRuntimeParameters runtime=(RTXDI_PTTemporalResamplingRuntimeParameters)0;
        runtime.pixelPosition=runtime.reservoirPosition=pixel;runtime.motionVector=float3(surface.motion.xy+(g.options.w?g.jitter.xy-g.previousJitter.xy:0),surface.motion.z-surface.positionDepth.w);
        runtime.cameraPos=g.eye.xyz;runtime.prevCameraPos=g.previousEye.xyz;runtime.prevPrevCameraPos=g.previousEye.xyz;
        RTXDI_RandomSamplerState rng=RTXDI_InitRandomSampler(pixel,g.frame.x,RTXDI_PT_TEMPORAL_RESAMPLING_RANDOM_SEED);
        bool selected=false;RAB_PathTracerUserData user=(RAB_PathTracerUserData)0;
        reservoir=RTXDI_PTTemporalResampling(p,runtime,PtShiftParameters(),PtReconnectionParameters(),RuntimeParameters(),ReservoirParameters(),rng,PtIndices(),selected,user);
    }
    RTXDI_StorePTReservoir(reservoir,ReservoirParameters(),pixel,3);
}
void PtSpatial(uint2 pixel) {
    RTXDI_PTReservoir reservoir=RTXDI_LoadPTReservoir(ReservoirParameters(),pixel,3);
    if(g.pathSampling.x && RAB_GetGBufferSurface(pixel,false).valid) {
        RTXDI_PTSpatialResamplingParameters p=(RTXDI_PTSpatialResamplingParameters)0;
        p.numSpatialSamples=g.pathSampling.x;p.numDisocclusionBoostSamples=g.pathSampling.y;
        p.maxTemporalHistory=g.sampling.z;p.samplingRadius=12;p.normalThreshold=.9;p.depthThreshold=.02;
        RTXDI_PTSpatialResamplingRuntimeParameters runtime=(RTXDI_PTSpatialResamplingRuntimeParameters)0;
        runtime.pixelPosition=runtime.reservoirPosition=pixel;runtime.viewportSize=g.size.xy;
        runtime.cameraPos=g.eye.xyz;runtime.prevCameraPos=g.previousEye.xyz;runtime.prevPrevCameraPos=g.previousEye.xyz;
        bool selected=false;RAB_PathTracerUserData user=(RAB_PathTracerUserData)0;
        RTXDI_RandomSamplerState rng=RTXDI_InitRandomSampler(pixel,g.frame.x,RTXDI_PT_SPATIAL_RESAMPLING_RANDOM_SEED);
        reservoir=RTXDI_PTSpatialResampling(runtime,p,PtShiftParameters(),PtReconnectionParameters(),ReservoirParameters(),PtIndices(),RuntimeParameters(),rng,selected,user);
    }
    RTXDI_StorePTReservoir(reservoir,ReservoirParameters(),pixel,g.frame.x&1);
    // Decorrelation changes shading only, never the reservoir persisted for reuse.
    RTXDI_PTDecorrelationParameters decorrelation=(RTXDI_PTDecorrelationParameters)0;
    decorrelation.decorrelationMode=RTXDI_PT_DECORRELATION_MODE_UNIFORM;decorrelation.decorrelationFactor=g.options.w?.1:0;
    RTXDI_PTApplyDecorrelation(pixel,pixel,reservoir,decorrelation,PtIndices(),ReservoirParameters(),g.frame.x);
    Surface s=RAB_GetGBufferSurface(pixel,false);float3 diffuse=0,specular=0;
    if(s.valid && reservoir.weightSum>0) {
        float3 direction=SafeNormal(reservoir.translatedWorldPosition-s.position,s.normal);bool delta=false;
        if(reservoir.rcVertexLength>2){RTXDI_RandomSamplerState rng=RTXDI_GetRngForShading(reservoir);RTXDI_BrdfRaySample sample=PtSampleBsdf(s,rng);direction=sample.outDirection;delta=sample.properties.IsDelta();}
        BsdfEval bsdf=EvaluateBsdf(s,direction);float3 fraction=delta?0:bsdf.diffuse/max(bsdf.diffuse+bsdf.specular,1e-20);
        float3 radiance=reservoir.targetFunction*reservoir.weightSum;
        radiance=select(isfinite(radiance),max(0,radiance),0);diffuse=radiance*fraction;specular=radiance-diffuse;
    }
    RWTexture2D<float4> raw=ResourceDescriptorHeap[g.guides.z],indirect=ResourceDescriptorHeap[g.signals.x];
    RWTexture2D<float4> diffuseOutput=ResourceDescriptorHeap[g.signals.y],specularOutput=ResourceDescriptorHeap[g.signals.z];
    float3 throughput=ReadSurface(pixel,false).throughput.xyz;diffuse*=throughput;specular*=throughput;
    raw[pixel]+=float4(diffuse+specular,0);indirect[pixel]=float4(diffuse+specular,1);
    diffuseOutput[pixel]+=float4(diffuse,0);specularOutput[pixel]+=float4(specular,0);
}
[shader("raygeneration")]
void RayGeneration() {
    uint2 pixel=DispatchRaysIndex().xy;
#if PT_PASS==0
    PtInitial(pixel);
#elif PT_PASS==1
    PtTemporal(pixel);
#else
    PtSpatial(pixel);
#endif
}
