#include "reconstruction.hlsli"
[numthreads(8,8,1)]
void PrepareGuides(uint3 id:SV_DispatchThreadID) {
    if(any(id.xy>=g.size.xy))return;SurfaceData s=ReadSurface(id.xy,false);
    RWTexture2D<float4> normals=ResourceDescriptorHeap[g.reconstruction.x],motion=ResourceDescriptorHeap[g.reconstruction.z];
    RWTexture2D<float> viewZ=ResourceDescriptorHeap[g.reconstruction.y];
    // NRD_NORMAL_ENCODING=RGBA16_SNORM and linear perceptual roughness.
    normals[id.xy]=float4(s.virtualNormal.xyz,s.normalRoughness.w);
    viewZ[id.xy]=s.identity.w?s.positionDepth.w:1e6;
    motion[id.xy]=float4(s.motion.xy,s.motion.z-s.positionDepth.w,0);
    RWTexture2D<float4> diffuse=ResourceDescriptorHeap[g.signals.y],specular=ResourceDescriptorHeap[g.signals.z];
    RWTexture2D<float4> demodDiffuse=ResourceDescriptorHeap[g.denoised.z],demodSpecular=ResourceDescriptorHeap[g.denoised.w];
    float4 d=diffuse[id.xy],r=specular[id.xy];
    demodDiffuse[id.xy]=float4(min(max(0,d.xyz-SurfaceEmission(s))/DiffuseFactor(s),65504),min(d.w,65504));
    demodSpecular[id.xy]=float4(min(max(0,r.xyz)/SpecularFactor(s),65504),min(r.w,65504));
    RWTexture2D<float4> diffuseAlbedo=ResourceDescriptorHeap[g.dlss.x],specularAlbedo=ResourceDescriptorHeap[g.dlss.y];
    RWTexture2D<float> specularDistance=ResourceDescriptorHeap[g.dlss.z],hardwareDepth=ResourceDescriptorHeap[g.dlssGuides.y];
    RWTexture2D<float2> dlssMotion=ResourceDescriptorHeap[g.dlssGuides.x];
    RWTexture2D<float2> specularMotion=ResourceDescriptorHeap[g.dlssGuides.z];
    diffuseAlbedo[id.xy]=float4(s.albedoOpacity.xyz*s.throughput.xyz,1);
    // Directional reflectance approximation, including grazing-angle Fresnel.
    float nv=saturate(dot(s.normalRoughness.xyz,s.viewPath.xyz));
    float roughness=s.normalRoughness.w;float4 coeff=roughness*float4(-1,-.0275,-.572,.022)+float4(1,.0425,1.04,-.04);
    float a004=min(coeff.x*coeff.x,exp2(-9.28*nv))*coeff.x+coeff.y;float2 ab=float2(-1.04,1.04)*a004+coeff.zw;
    specularAlbedo[id.xy]=float4(max(0,s.specularDistance.xyz*ab.x+ab.y)*s.throughput.xyz,1);
    specularDistance[id.xy]=r.w;dlssMotion[id.xy]=s.motion.xy;
    specularMotion[id.xy]=s.specularMotion.xy;
    hardwareDepth[id.xy]=s.identity.w?saturate(1-.05/max(s.positionDepth.w,.05)):1;
}
[numthreads(8,8,1)]
void ComposeForDlss(uint3 id:SV_DispatchThreadID) {
    if(any(id.xy>=g.size.xy))return;RWTexture2D<float4> color=ResourceDescriptorHeap[g.dlss.w];color[id.xy]=float4(ReconstructedColor(id.xy),1);
}
