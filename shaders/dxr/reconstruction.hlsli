#ifndef RENDERER_DXR_RECONSTRUCTION
#define RENDERER_DXR_RECONSTRUCTION
#include "scene.hlsli"
#include "NRD.hlsli"
void MaterialFactors(SurfaceData surface,out float3 diffuse,out float3 specular) {
    // Use the same directional BRDF factors on both sides of RELAX. F0 alone
    // leaves grazing-angle and roughness variation in the denoised signal.
    NRD_MaterialFactors(surface.normalRoughness.xyz,surface.viewPath.xyz,
        surface.albedoOpacity.xyz,surface.specularDistance.xyz,surface.normalRoughness.w,diffuse,specular);
    diffuse=max(diffuse*surface.throughput.xyz,.001);
    specular=max(specular*surface.throughput.xyz,.001);
}
float3 DiffuseFactor(SurfaceData surface){float3 d,s;MaterialFactors(surface,d,s);return d;}
float3 SpecularFactor(SurfaceData surface){float3 d,s;MaterialFactors(surface,d,s);return s;}
float3 SurfaceEmission(SurfaceData surface){return surface.emission.xyz*surface.throughput.xyz;}
float3 ReconstructedColor(int2 pixel) {
    pixel=clamp(pixel,0,int2(g.size.xy)-1);SurfaceData s=ReadSurface(pixel,false);
    RWTexture2D<float4> raw=ResourceDescriptorHeap[g.guides.z];
    if(g.options.w==0 || g.options.w==3 || s.identity.w==0)return raw[pixel].xyz;
    Texture2D<float4> diffuse=ResourceDescriptorHeap[g.denoised.x],specular=ResourceDescriptorHeap[g.denoised.y];
    return max(0,diffuse[pixel].xyz*DiffuseFactor(s)+specular[pixel].xyz*SpecularFactor(s)+SurfaceEmission(s));
}
float3 ToYCoCg(float3 c){return float3(dot(c,float3(.25,.5,.25)),dot(c,float3(.5,0,-.5)),dot(c,float3(-.25,.5,-.25)));}
float3 FromYCoCg(float3 c){return float3(c.x+c.y-c.z,c.x+c.z,c.x-c.y-c.z);}
#endif
