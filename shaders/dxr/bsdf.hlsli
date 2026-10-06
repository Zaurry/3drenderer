#ifndef RENDERER_DXR_BSDF
#define RENDERER_DXR_BSDF
#include "scene.hlsli"
// Dynamic indexing of a static HLSL array expands into expensive selection
// code. Keep the same full-precision values/interpolation in a read-only SRV.
float Energy(float nv,float roughness){StructuredBuffer<float> lut=ResourceDescriptorHeap[g.bsdfTables.x];float2 xy=saturate(float2(nv,roughness))*31;uint2 p=min(uint2(xy),30);float2 t=xy-p;return lerp(lerp(lut[p.y*32+p.x],lut[p.y*32+p.x+1],t.x),lerp(lut[(p.y+1)*32+p.x],lut[(p.y+1)*32+p.x+1],t.x),t.y);}
float AverageEnergy(float r){StructuredBuffer<float> lut=ResourceDescriptorHeap[g.bsdfTables.x];float x=saturate(r)*31;uint p=min(uint(x),30);return lerp(lut[1024+p],lut[1024+p+1],x-p);}
float3 Fresnel(float c,float3 f0,float3 f90){return f0+(f90-f0)*pow(1-saturate(c),5);}
float G1(float c,float a){if(c<=0)return 0;return 2/(1+sqrt(1+a*a*max(0,1-c*c)/max(c*c,1e-12)));}
float Lambda(float c,float a){return .5*(sqrt(1+a*a*max(0,1-c*c)/max(c*c,1e-12))-1);}
float Distribution(float nh,float a){float q=(1-nh)*(1+nh)+a*a*nh*nh;return a*a/(PI*q*q);}
float SpecularProbability(Surface s){float e=AverageEnergy(s.roughness);float3 f=s.f0+(s.f90-s.f0)/21;float spec=Luminance(f*e);float broad=Luminance(s.diffuse+f*f*(e*(1-e))/max(1-f*(1-e),1e-6));return clamp(spec/max(spec+broad,1e-8),.05,.95);}
struct BsdfEval {float3 diffuse,specular;float pdf;};
bool IsDeltaSurface(Surface s){return s.type==2 || (s.type==1 && LoadMaterial(s.material).emissionRoughness.w<=0);}
BsdfEval EvaluateBsdf(Surface s,float3 l) {
    BsdfEval e=(BsdfEval)0;
    if(s.type==0){e.diffuse=s.base*INV_PI;e.pdf=saturate(dot(s.normal,l))*INV_PI;return e;}
    if(s.type==3 || IsDeltaSurface(s))return e;
    float nv=saturate(dot(s.normal,s.view)),nl=saturate(dot(s.normal,l));if(nv<=0 || nl<=0 || dot(s.view+l,s.view+l)<1e-20 || s.type==2)return e;
    float3 h=normalize(s.view+l);float nh=saturate(dot(s.normal,h)),vh=saturate(dot(s.view,h)),a=s.roughness*s.roughness;
    float d=Distribution(nh,a),visibility=1/(1+Lambda(nv,a)+Lambda(nl,a));
    e.specular=Fresnel(vh,s.f0,s.f90)*(d*visibility/max(4*nv*nl,1e-12));
    float avg=AverageEnergy(s.roughness),missing=1-avg;float3 f=s.f0+(s.f90-s.f0)/21;
    if(missing>1e-6)e.specular+=f*f*avg/max(1-f*missing,1e-6)*((1-Energy(nv,s.roughness))*(1-Energy(nl,s.roughness))/(PI*missing));
    float3 df=Fresnel(vh,s.diffuseF0,s.diffuseF90);e.diffuse=s.diffuse*(1-(s.diffuseMax?MaxComponent(df).xxx:df))*INV_PI;
    float probability=SpecularProbability(s);e.pdf=lerp(nl*INV_PI,d*G1(nv,a)/max(4*nv,1e-12),probability);return e;
}
float3 VisibleGgx(float3 v,float a,float2 u) {
    float3 vh=normalize(float3(a*v.xy,v.z));float len=dot(vh.xy,vh.xy);
    float3 t1=len>1e-12?float3(-vh.y,vh.x,0)/sqrt(len):float3(1,0,0),t2=cross(vh,t1);
    float radius=sqrt(u.x),phi=2*PI*u.y;float x=radius*cos(phi),y=radius*sin(phi),blend=.5*(1+vh.z);
    y=(1-blend)*sqrt(max(0,1-x*x))+blend*y;float3 nh=x*t1+y*t2+sqrt(max(0,1-x*x-y*y))*vh;
    return normalize(float3(a*nh.xy,max(0,nh.z)));
}
float DielectricFresnel(float cosine,float eta) {
    float sinT2=eta*eta*max(0,1-cosine*cosine);if(sinT2>=1)return 1;
    float ct=sqrt(1-sinT2);float rs=(eta*cosine-ct)/(eta*cosine+ct);float rp=(cosine-eta*ct)/(cosine+eta*ct);return .5*(rs*rs+rp*rp);
}
bool SampleBsdf(Surface s,inout uint rng,out float3 direction,out float3 weight,out float pdf,out bool delta,out bool specularLobe) {
    weight=0;pdf=0;delta=false;direction=0;specularLobe=false;
    if(s.type==3)return false;
    if(s.type==2) {
        float eta=s.front?1/s.ior:s.ior,cosine=saturate(dot(s.view,s.geometric)),f=DielectricFresnel(cosine,eta);
        if(Random(rng)<f){direction=reflect(-s.view,s.geometric);weight=1;}
        else {direction=refract(-s.view,s.geometric,eta);weight=s.base;}
        delta=true;specularLobe=true;pdf=1;return dot(direction,direction)>0;
    }
    Material m=LoadMaterial(s.material);
    if(s.type==1 && m.emissionRoughness.w<=0){direction=reflect(-s.view,s.normal);weight=s.base;pdf=1;delta=true;specularLobe=true;return dot(direction,s.geometric)>0;}
    float3 t,b;Basis(s.normal,t,b);float choose=Random(rng);float2 u=float2(Random(rng),Random(rng));
    if(s.type!=0 && choose<SpecularProbability(s)) {
        specularLobe=true;
        float3 local=VisibleGgx(float3(dot(s.view,t),dot(s.view,b),dot(s.view,s.normal)),s.roughness*s.roughness,u);
        direction=reflect(-s.view,normalize(t*local.x+b*local.y+s.normal*local.z));
    } else {float radius=sqrt(u.x),phi=2*PI*u.y;direction=t*(radius*cos(phi))+b*(radius*sin(phi))+s.normal*sqrt(max(0,1-u.x));}
    BsdfEval e=EvaluateBsdf(s,direction);pdf=e.pdf;
    if(pdf<=0 || dot(direction,s.geometric)<=0)return false;
    weight=(e.diffuse+e.specular)*saturate(dot(s.normal,direction))/pdf;
    return all(isfinite(weight));
}
bool SampleBsdf(Surface s,inout uint rng,out float3 direction,out float3 weight,out float pdf,out bool delta) {
    bool specularLobe;return SampleBsdf(s,rng,direction,weight,pdf,delta,specularLobe);
}
float Mis(float a,float b){return a*a/max(a*a+b*b,1e-30);}
#endif
