#ifndef RENDERER_DXR_MATERIAL_RESOLVE
#define RENDERER_DXR_MATERIAL_RESOLVE
#include "reconstruction.hlsli"

bool IntersectKnownTriangle(Instance instance,uint primitive,float3 origin,float3 direction,out float2 bary,out float distance) {
    Vertex a,b,c;LoadVertices(instance,primitive,a,b,c);
    float3 e1=b.position.xyz-a.position.xyz,e2=c.position.xyz-a.position.xyz;
    float3 p=cross(direction,e2);float determinant=dot(e1,p);
    bary=0;distance=0;
    if(abs(determinant)<1e-8*max(length(e1)*length(p),1e-20))return false;
    float3 offset=origin-a.position.xyz,q=cross(offset,e1);
    bary=float2(dot(offset,p),dot(direction,q))/determinant;
    distance=dot(e2,q)/determinant;
    return all(bary>=0) && bary.x+bary.y<=1 && distance>0;
}

// Re-evaluate material colors on a known, continuous primary surface. Lighting
// remains at the tracing resolution; its demodulated signals are combined
// with material colors at the output sample. This needs no extra BVH
// traversal and never extrapolates across silhouettes, alpha or PSR/glass.
bool ResolveMaterialDetail(uint2 pixel,out float3 color,out SurfaceData surface) {
    color=0;surface=(SurfaceData)0;
    if(!g.pathSampling.w || g.options.w!=1 || all(g.size.xy==g.size.zw))return false;
    float2 uv=(float2(pixel)+.5+g.jitter.xy)/g.size.zw;
    // Sample illumination on the unjittered output grid. Only the material
    // sample uses output-pixel jitter, otherwise lighting is blurred twice.
    float2 source=(float2(pixel)+.5)*float2(g.size.xy)/g.size.zw-.5-g.jitter.xy;
    int2 base=int2(floor(source));float2 blend=frac(source);
    if(any(base<0) || any(base+1>=int2(g.size.xy)))return false;
    SurfaceData center=ReadSurface(base,false);
    if(!center.identity.w || center.viewPath.w>0)return false;
    Material material=LoadMaterial(center.identity.z);
    if(material.flags.x!=0 || material.pbr.w==2 || (material.pbr.w==1 && material.emissionRoughness.w<=0))return false;
    bool textured=false;
    [unroll]for(uint slot=0;slot<11;++slot)textured=textured || material.textures[slot].texture!=0;
    if(!textured)return false;
    Instance instance=LoadInstance(center.identity.x);
    float3 direction=normalize(g.forward.xyz+g.right.xyz*((uv.x-.5)*g.right.w)+g.up.xyz*((.5-uv.y)*g.up.w));
    float3 localOrigin=TransformPoint(instance.inverse,g.eye.xyz),localDirection=TransformDirection(instance.inverse,direction);
    float closest=1e30;uint primitive=0xffffffff;float2 bary=0;
    float planeTolerance=max(.001,.005*center.positionDepth.w);
    [unroll]for(int y=0;y<2;++y)[unroll]for(int x=0;x<2;++x) {
        SurfaceData neighbor=ReadSurface(base+int2(x,y),false);
        // All contributors must agree on a continuous opaque surface. In
        // particular, a foreground sample cannot replace background coverage.
        if(!neighbor.identity.w || any(neighbor.identity.xz!=center.identity.xz) || neighbor.viewPath.w>0 ||
            dot(neighbor.geometricMetallic.xyz,center.geometricMetallic.xyz)<.95 ||
            abs(dot(neighbor.positionDepth.xyz-center.positionDepth.xyz,center.geometricMetallic.xyz))>planeTolerance)return false;
        float2 candidateBary;float candidateDistance;
        if(IntersectKnownTriangle(instance,neighbor.identity.y,localOrigin,localDirection,candidateBary,candidateDistance) && candidateDistance<closest) {
            closest=candidateDistance;bary=candidateBary;primitive=neighbor.identity.y;
        }
    }
    if(primitive==0xffffffff)return false;
    surface=PackSurface(EvaluateSurface(center.identity.x,primitive,bary,closest,direction));
    Texture2D<float4> diffuse=ResourceDescriptorHeap[g.denoised.x],specular=ResourceDescriptorHeap[g.denoised.y];
    [unroll]for(int y=0;y<2;++y)[unroll]for(int x=0;x<2;++x) {
        int2 p=base+int2(x,y);float weight=(x?blend.x:1-blend.x)*(y?blend.y:1-blend.y);
        SurfaceData neighbor=ReadSurface(p,false),materialSample=surface;
        // Scalar irradiance cannot recover a different normal's reflection
        // direction. Preserve the sampled lobe while recovering albedo/F0
        // texture detail; replacing its normal here can invent highlights.
        materialSample.normalRoughness=neighbor.normalRoughness;
        materialSample.viewPath=neighbor.viewPath;
        float3 diffuseFactor,specularFactor;MaterialFactors(materialSample,diffuseFactor,specularFactor);
        color+=(diffuse[p].xyz*diffuseFactor+specular[p].xyz*specularFactor)*weight;
    }
    color=max(0,color+SurfaceEmission(surface));return true;
}
#endif
