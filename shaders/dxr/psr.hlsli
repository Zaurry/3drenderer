#ifndef RENDERER_DXR_PSR
#define RENDERER_DXR_PSR
#include "trace.hlsli"

float3 PreviousPosition(Surface s) {
    Instance i=LoadInstance(s.instance);return TransformPoint(i.previous,TransformPoint(i.inverse,s.position));
}
float3 PreviousNormal(Surface s) {
    Instance i=LoadInstance(s.instance);
    float3 local=i.world[0].xyz*s.normal.x+i.world[1].xyz*s.normal.y+i.world[2].xyz*s.normal.z;
    float3 x=cross(i.previous[1].xyz,i.previous[2].xyz),y=cross(i.previous[2].xyz,i.previous[0].xyz),z=cross(i.previous[0].xyz,i.previous[1].xyz);
    float determinant=dot(i.previous[0].xyz,x);
    return SafeNormal(float3(dot(x,local),dot(y,local),dot(z,local))/determinant,s.normal);
}
float4x4 ReflectionPlane(float3 n,float3 p) {
    float4x4 r=float4x4(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1);
    [unroll]for(uint j=0;j<3;++j){r[j].xyz-=2*n[j]*n;r[j].w=2*n[j]*dot(n,p);}
    return r;
}
bool PsrEligible(Surface s) {
    // Delta metal with an unperturbed planar normal. Curved/normal-mapped
    // mirrors need curvature-aware tracking and use the regular path instead.
    if(!(g.sampling.w&2) || s.type!=1 || LoadMaterial(s.material).emissionRoughness.w>0 || any(s.emission>0))return false;
    if(dot(s.normal,s.geometric)<.9999)return false;
    Instance i=LoadInstance(s.instance);Vertex a,b,c;LoadVertices(i,s.primitive,a,b,c);
    return abs(dot(TransformNormal(i,a.normal.xyz),s.geometric))>.9999 && abs(dot(TransformNormal(i,b.normal.xyz),s.geometric))>.9999 && abs(dot(TransformNormal(i,c.normal.xyz),s.geometric))>.9999;
}
struct PrimaryPath {
    Hit hit;float3 origin,direction,throughput;
    float4x4 reflection,previousReflection;uint depth,chain;
};
PrimaryPath TracePrimary(float3 origin,float3 direction,inout uint rng) {
    PrimaryPath p=(PrimaryPath)0;p.origin=origin;p.direction=direction;p.throughput=1;
    p.reflection=p.previousReflection=float4x4(1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1);
    [loop]for(uint bounce=0;bounce<g.frame.z;++bounce) {
        p.hit=Trace(p.origin,p.direction,rng,1e30,bounce>0);
        if(p.hit.instance==0xffffffff)break;
        // Most surfaces cannot participate in a delta mirror chain. Inspect the
        // material first; evaluating every textured surface here would repeat
        // all texture/normal work in the integrator below.
        if(bounce+1==g.frame.z || !(g.sampling.w&2))break;
        Instance instance=LoadInstance(p.hit.instance);Material material=LoadMaterial(MaterialIndex(instance,p.hit.primitive));
        if((uint)material.pbr.w!=1 || material.emissionRoughness.w>0)break;
        Surface s=EvaluateSurface(p.hit.instance,p.hit.primitive,p.hit.bary,p.hit.t,p.direction);
        if(bounce+1==g.frame.z || !PsrEligible(s))break;
        p.reflection=mul(p.reflection,ReflectionPlane(s.normal,s.position));
        p.previousReflection=mul(p.previousReflection,ReflectionPlane(PreviousNormal(s),PreviousPosition(s)));
        p.throughput*=s.base;p.chain=Hash(p.chain^Hash(s.instance+1)^Hash(s.primitive+1));++p.depth;
        p.direction=reflect(p.direction,s.normal);p.origin=Offset(s.position,s.geometric,p.direction);
    }
    return p;
}
SurfaceData PackPrimary(Surface s,PrimaryPath path) {
    SurfaceData d=PackSurface(s);d.viewPath.w=path.depth;d.throughput=float4(path.throughput,1);
    if(path.depth) {
        float3 virtualPosition=mul(path.reflection,float4(s.position,1)).xyz;
        float3 previousPosition=mul(path.previousReflection,float4(PreviousPosition(s),1)).xyz;
        d.positionDepth.w=dot(virtualPosition-g.eye.xyz,g.forward.xyz);
        d.virtualNormal=float4(normalize(mul((float3x3)path.reflection,s.normal)),asfloat(path.chain));
        d.motion.xyz=float3(Project(previousPosition,true)-Project(virtualPosition,false),dot(previousPosition-g.previousEye.xyz,g.previousForward.xyz));
        d.specularMotion=d.motion;
    }
    return d;
}
float4 ReflectionMotion(Surface s,inout uint rng) {
    if(s.type==2 || s.roughness>.3)return float4(0,0,0,0);
    float3 direction=reflect(-s.view,s.normal);Hit h=Trace(Offset(s.position,s.geometric,direction),direction,rng);
    if(h.instance==0xffffffff)return float4(0,0,0,0);
    // Motion needs only hit position and instance transforms, never the
    // secondary material's complete texture/BRDF evaluation.
    float3 secondaryPosition=Offset(s.position,s.geometric,direction)+direction*h.t;
    Instance secondary=LoadInstance(h.instance);
    float3 oldPosition=TransformPoint(secondary.previous,TransformPoint(secondary.inverse,secondaryPosition));
    float3 position=mul(ReflectionPlane(s.normal,s.position),float4(secondaryPosition,1)).xyz;
    float3 previous=mul(ReflectionPlane(PreviousNormal(s),PreviousPosition(s)),float4(oldPosition,1)).xyz;
    return float4(Project(previous,true)-Project(position,false),dot(previous-g.previousEye.xyz,g.previousForward.xyz),1);
}
#endif
