#include "lighting.hlsli"
#include "psr.hlsli"
struct PathResult {float3 radiance,direct,indirect,diffuse,specular;float diffuseDistance,specularDistance;SurfaceData surface;};
PathResult Integrate(float3 origin,float3 direction,inout uint rng) {
    PathResult result=(PathResult)0;PrimaryPath primary=TracePrimary(origin,direction,rng);
    origin=primary.origin;direction=primary.direction;
    float3 throughput=primary.throughput,firstDiffuse=1,firstSpecular=0;
    Surface previous=EmptySurface();float previousPdf=0;bool previousDelta=true,firstSpecularLobe=false;
    [loop]for(uint bounce=0;bounce<g.frame.z-primary.depth;++bounce) {
        Hit hit=primary.hit;if(bounce)hit=Trace(origin,direction,rng);
        if(bounce==1){float distance=hit.instance==0xffffffff?100000:hit.t;if(firstSpecularLobe)result.specularDistance=distance;else result.diffuseDistance=distance;}
        if(hit.instance==0xffffffff) {
            float weight=previousDelta?1:Mis(previousPdf,SelectionPdf(g.lightExtra.w)*EnvironmentPdf(direction));
            float3 energy=throughput*Environment(direction)*weight;
            if(bounce==0){if(g.environmentInfo.y!=0 || primary.depth){result.radiance+=energy;result.direct+=energy;result.diffuse+=energy;}}
            else {result.radiance+=energy;result.indirect+=energy;result.diffuse+=energy*firstDiffuse;result.specular+=energy*firstSpecular;}
            break;
        }
        Surface s=EvaluateSurface(hit.instance,hit.primitive,hit.bary,hit.t,direction);
        if(bounce==0){result.surface=PackPrimary(s,primary);if(!primary.depth && (g.sampling.w&2) && (g.options.w==3 || g.options.z==8)){uint motionRng=rng;float4 motion=ReflectionMotion(s,motionRng);if(motion.w)result.surface.specularMotion=motion;}}
        if(MaxComponent(s.emission)>0) {
            float weight=previousDelta?1:Mis(previousPdf,HitLightPdf(s,previous,direction));
            float3 energy=throughput*s.emission*weight;
            result.radiance+=energy;if(bounce==0){result.direct+=energy;result.diffuse+=energy;}else{result.indirect+=energy;result.diffuse+=energy*firstDiffuse;result.specular+=energy*firstSpecular;}
        }
        if(s.type==3)break;
        float3 directDiffuse=0,directSpecular=0;
        if(bounce>0 || !g.options.x)DirectLighting(s,rng,directDiffuse,directSpecular,bounce+1<g.frame.z-primary.depth);
        float3 direct=throughput*(directDiffuse+directSpecular);result.radiance+=direct;
        if(bounce==0){result.direct+=direct;result.diffuse+=throughput*directDiffuse;result.specular+=throughput*directSpecular;}
        else {result.indirect+=direct;result.diffuse+=direct*firstDiffuse;result.specular+=direct*firstSpecular;}
        if(bounce+1==g.frame.z-primary.depth || g.options.y)break;
        float3 weight;bool delta,specularLobe;float pdf;
        if(!SampleBsdf(s,rng,direction,weight,pdf,delta,specularLobe))break;
        if(bounce==0){firstSpecularLobe=specularLobe;BsdfEval e=EvaluateBsdf(s,direction);firstDiffuse=delta?0:e.diffuse/max(e.diffuse+e.specular,1e-20);firstSpecular=1-firstDiffuse;}
        throughput*=weight;previous=s;previousPdf=pdf;previousDelta=delta;
        if(bounce>=3){float survival=clamp(MaxComponent(throughput),.05,.95);if(Random(rng)>survival)break;throughput/=survival;}
        origin=Offset(s.position,s.geometric,direction);
    }
    return result;
}
[shader("raygeneration")]
void RayGeneration() {
    uint2 pixel=DispatchRaysIndex().xy;uint rng=Hash(pixel.x+pixel.y*g.size.x)^Hash(g.frame.x+1);
    PathResult sum=(PathResult)0;
    for(uint sample=0;sample<g.frame.y;++sample) {
        float2 jitter=g.options.w==0?float2(Random(rng),Random(rng)):g.jitter.xy+.5;
        PathResult p=Integrate(g.eye.xyz,CameraDirection(float2(pixel)+jitter),rng);
        if(sample==0){sum.surface=p.surface;sum.diffuseDistance=p.diffuseDistance;sum.specularDistance=p.specularDistance;}
        sum.radiance+=p.radiance;sum.direct+=p.direct;sum.indirect+=p.indirect;sum.diffuse+=p.diffuse;sum.specular+=p.specular;
    }
    float scale=1.0/g.frame.y;
    RWStructuredBuffer<SurfaceData> surfaces=ResourceDescriptorHeap[g.guides.x];surfaces[pixel.y*g.size.x+pixel.x]=sum.surface;
    RWTexture2D<float4> raw=ResourceDescriptorHeap[g.guides.z],direct=ResourceDescriptorHeap[g.guides.w],indirect=ResourceDescriptorHeap[g.signals.x];
    RWTexture2D<float4> diffuse=ResourceDescriptorHeap[g.signals.y],specular=ResourceDescriptorHeap[g.signals.z];
    raw[pixel]=float4(max(0,sum.radiance*scale),1);direct[pixel]=float4(max(0,sum.direct*scale),1);indirect[pixel]=float4(max(0,sum.indirect*scale),1);
    diffuse[pixel]=float4(max(0,sum.diffuse*scale),sum.diffuseDistance);specular[pixel]=float4(max(0,sum.specular*scale),sum.specularDistance);
}
