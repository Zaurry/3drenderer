#ifndef RENDERER_DXR_SCENE
#define RENDERER_DXR_SCENE
#define PI 3.14159265358979323846
#define INV_PI 0.31830988618379067154
struct Vertex {float4 position,normal,uv,tangent,color;};
struct Instance {float4 world[3],inverse[3],previous[3];uint4 geometry,identity;};
struct TextureBinding {uint texture,sampler,texcoord;float rotation;float4 transform;};
struct Material {float4 baseOpacity,emissionRoughness,specularMetallic,optics,pbr;uint4 flags;TextureBinding textures[11];};
struct Light {float4 positionType,directionRadius,radianceRange,axisUInner,axisVOuter;uint4 identity;};
struct SurfaceData {float4 positionDepth,normalRoughness,geometricMetallic,albedoOpacity,specularDistance,motion,emission;uint4 identity;float4 base,f90,diffuseF0,viewPath,virtualNormal,throughput,specularMotion;};
struct FrameConstants {
    float4 eye,forward,right,up,previousEye,previousForward,previousRight,previousUp;
    float4 environment,jitter,previousJitter;uint4 size,frame,scene,lighting,lightExtra,guides,signals,history,options,sampling,pathSampling,reconstruction,denoised,lightHistory,reservoirs,dlss,dlssGuides,bsdfTables;float4 environmentInfo;
};
ConstantBuffer<FrameConstants> g : register(b0);
RaytracingAccelerationStructure sceneAS : register(t0);
uint Hash(uint x){x^=x>>16;x*=0x7feb352d;x^=x>>15;x*=0x846ca68b;x^=x>>16;return x;}
float Random(inout uint state){state=Hash(state+0x9e3779b9);return float(state>>8)*(1.0/16777216.0);}
float Luminance(float3 c){return dot(c,float3(.2126,.7152,.0722));}
float MaxComponent(float3 c){return max(c.x,max(c.y,c.z));}
float3 SafeNormal(float3 n,float3 fallback){return dot(n,n)>1e-20?normalize(n):fallback;}
float3 TransformPoint(float4 m[3],float3 p){return float3(dot(m[0],float4(p,1)),dot(m[1],float4(p,1)),dot(m[2],float4(p,1)));}
float3 TransformDirection(float4 m[3],float3 p){return float3(dot(m[0].xyz,p),dot(m[1].xyz,p),dot(m[2].xyz,p));}
float3 TransformNormal(Instance i,float3 n){return SafeNormal(i.inverse[0].xyz*n.x+i.inverse[1].xyz*n.y+i.inverse[2].xyz*n.z,float3(0,1,0));}
void Basis(float3 n,out float3 t,out float3 b){t=normalize(cross(abs(n.z)<.999?float3(0,0,1):float3(1,0,0),n));b=cross(n,t);}
float3 Offset(float3 p,float3 n,float3 dir){return p+n*(dot(n,dir)>=0?1:-1)*max(1,MaxComponent(abs(p)))*3.814697266e-6;}
Instance LoadInstance(uint id){StructuredBuffer<Instance> b=ResourceDescriptorHeap[g.scene.z];return b[id];}
Material LoadMaterial(uint id){StructuredBuffer<Material> b=ResourceDescriptorHeap[g.scene.w];return b[id];}
uint MaterialIndex(Instance inst,uint primitive){StructuredBuffer<uint> slots=ResourceDescriptorHeap[g.scene.y];uint slot=slots[inst.geometry.y+primitive];return slot==0xffffffff?inst.identity.w:inst.geometry.z+slot;}
void LoadVertices(Instance inst,uint primitive,out Vertex a,out Vertex b,out Vertex c){StructuredBuffer<Vertex> vertices=ResourceDescriptorHeap[g.scene.x];uint i=inst.geometry.x+primitive*3;a=vertices[i];b=vertices[i+1];c=vertices[i+2];}
float2 MaterialUV(TextureBinding t,float4 uv){float2 q=(t.texcoord&1)?uv.zw:uv.xy;q*=t.transform.zw;float sn,cs;sincos(t.rotation,sn,cs);return float2(cs*q.x-sn*q.y,sn*q.x+cs*q.y)+t.transform.xy;}
float2 TextureUV(TextureBinding t,float4 uv){float2 q=MaterialUV(t,uv);if(t.texcoord&2)q.y=1-q.y;return q;}
float4 SampleTexture(TextureBinding t,float4 uv,float4 fallback){if(t.texture==0)return fallback;Texture2D<float4> tex=ResourceDescriptorHeap[NonUniformResourceIndex(t.texture)];SamplerState s=SamplerDescriptorHeap[NonUniformResourceIndex(t.sampler)];return tex.SampleLevel(s,TextureUV(t,uv),0);}
float SurfaceOpacity(Material m,float4 uv,float alpha){
    if((uint)m.pbr.w==4 && m.flags.x==0)return 1;
    float opacity=m.baseOpacity.w*alpha;
    opacity*=Luminance(SampleTexture(m.textures[1],uv,1).xyz);
    opacity*=SampleTexture(m.textures[3],uv,1).w;
    return saturate(opacity);
}
struct Surface {
    float3 position,normal,geometric,view,base,diffuse,f0,f90,diffuseF0,diffuseF90,emission;
    float roughness,metallic,opacity,ior,depth,hitT;
    uint type,material,instance,primitive,psrDepth,psrChain;bool front,diffuseMax,valid;
};
Surface EmptySurface(){Surface s=(Surface)0;s.instance=0xffffffff;s.primitive=0xffffffff;return s;}
Surface EvaluateSurface(uint instance,uint primitive,float2 bary,float hitT,float3 rayDir) {
    Surface s=EmptySurface();s.valid=true;s.instance=instance;s.primitive=primitive;s.hitT=hitT;
    Instance inst=LoadInstance(instance);Vertex a,b,c;LoadVertices(inst,primitive,a,b,c);
    float3 weights=float3(1-bary.x-bary.y,bary);float3 local=a.position.xyz*weights.x+b.position.xyz*weights.y+c.position.xyz*weights.z;
    s.position=TransformPoint(inst.world,local);s.view=-rayDir;s.depth=dot(s.position-g.eye.xyz,g.forward.xyz);
    float3 geometric=TransformNormal(inst,cross(b.position.xyz-a.position.xyz,c.position.xyz-a.position.xyz));
    s.front=dot(rayDir,geometric)<0;s.geometric=s.front?geometric:-geometric;
    float3 n=TransformNormal(inst,a.normal.xyz*weights.x+b.normal.xyz*weights.y+c.normal.xyz*weights.z);
    if(dot(n,geometric)<0)n=-n;s.normal=s.front?n:-n;
    float4 uv=a.uv*weights.x+b.uv*weights.y+c.uv*weights.z;
    float4 color=a.color*weights.x+b.color*weights.y+c.color*weights.z;
    s.material=MaterialIndex(inst,primitive);Material m=LoadMaterial(s.material);s.type=(uint)m.pbr.w;
    s.base=max(0,m.baseOpacity.xyz*color.xyz*SampleTexture(m.textures[m.textures[3].texture?3:0],uv,1).xyz);
    s.emission=max(0,m.emissionRoughness.xyz*SampleTexture(m.textures[7],uv,1).xyz);
    s.opacity=SurfaceOpacity(m,uv,color.w);s.ior=max(1,m.optics.x);
    s.metallic=s.type==1?1:saturate(m.specularMetallic.w);
    s.roughness=s.type==0?1:clamp(m.emissionRoughness.w,.02,1);
    s.diffuseMax=true;
    if(s.type==4 && m.flags.z==1) {
        float4 sg=SampleTexture(m.textures[10],uv,1);
        s.f0=saturate(m.specularMetallic.xyz*sg.xyz);s.f90=1;
        s.diffuseF0=s.f0;s.diffuseF90=1;s.diffuseMax=false;
        s.diffuse=s.base*(1-MaxComponent(s.f0));s.metallic=0;s.roughness=clamp(1-m.pbr.y*sg.w,.02,1);
    } else {
        float4 mr=SampleTexture(m.textures[4],uv,1);s.roughness=clamp(s.roughness*mr.y,.02,1);s.metallic=saturate(s.metallic*mr.z);
        float strength=saturate(m.pbr.x)*SampleTexture(m.textures[8],uv,1).w;
        float3 tint=max(0,m.specularMetallic.xyz*SampleTexture(m.textures[9],uv,1).xyz);
        float ratio=(s.ior-1)/(s.ior+1);s.diffuseF0=saturate(tint*ratio*ratio)*strength;s.diffuseF90=strength;
        s.f0=lerp(s.diffuseF0,s.base,s.metallic);s.f90=lerp(strength.xxx,1.xxx,s.metallic);s.diffuse=s.base*(1-s.metallic);
    }
    if(m.textures[5].texture || m.textures[2].texture) {
        bool normalMap=m.textures[5].texture!=0;TextureBinding binding=m.textures[normalMap?5:2];
        // Derive the frame from the selected, transformed UV set, just as RTRT
        // does. Image storage's vertical flip is not a material UV transform.
        float2 uv0=MaterialUV(binding,a.uv),d1=MaterialUV(binding,b.uv)-uv0,d2=MaterialUV(binding,c.uv)-uv0;
        float determinant=d1.x*d2.y-d1.y*d2.x;
        if(abs(determinant)>1e-12) {
            float3 e1=TransformDirection(inst.world,b.position.xyz-a.position.xyz),e2=TransformDirection(inst.world,c.position.xyz-a.position.xyz);
            float3 tangent=(e1*d2.y-e2*d1.y)/determinant,rawBitangent=(e2*d1.x-e1*d2.x)/determinant;
            tangent-=s.normal*dot(tangent,s.normal);
            if(dot(tangent,tangent)>1e-24) {
                tangent=normalize(tangent);float3 bitangent=normalize(cross(s.normal,tangent));
                if(dot(bitangent,rawBitangent)<0)bitangent=-bitangent;
                float3 mapped=float3(0,0,1);
                if(normalMap){mapped=SampleTexture(binding,uv,float4(.5,.5,1,1)).xyz*2-1;mapped.xy*=m.optics.w;}
                else {
                    Texture2D<float4> tex=ResourceDescriptorHeap[NonUniformResourceIndex(binding.texture)];uint w,h;tex.GetDimensions(w,h);
                    float4 du=float4(1.0/w,0,0,0),dv=float4(0,1.0/h,0,0);
                    float dx=Luminance(SampleTexture(binding,uv+du,0).xyz-SampleTexture(binding,uv-du,0).xyz)*.5;
                    float dy=Luminance(SampleTexture(binding,uv+dv,0).xyz-SampleTexture(binding,uv-dv,0).xyz)*.5;
                    // Existing OBJ bump scale is per texel, not per UV unit.
                    mapped=float3(-dx*m.optics.z,-dy*m.optics.z,1);
                }
                float3 changed=SafeNormal(tangent*mapped.x+bitangent*mapped.y+s.normal*mapped.z,s.normal);
                s.normal=dot(changed,s.geometric)<0?-changed:changed;
            }
        }
    }
    return s;
}
float2 Project(float3 p,bool previous) {
    float3 v=p-(previous?g.previousEye.xyz:g.eye.xyz);
    float4 f=previous?g.previousForward:g.forward,r=previous?g.previousRight:g.right,u=previous?g.previousUp:g.up;
    float z=dot(v,f.xyz);return float2(.5+dot(v,r.xyz)/(max(z,1e-6)*r.w),.5-dot(v,u.xyz)/(max(z,1e-6)*u.w))*g.size.xy;
}
SurfaceData PackSurface(Surface s) {
    SurfaceData d=(SurfaceData)0;d.identity=uint4(s.instance,s.primitive,s.material,s.valid?1:0);
    if(!s.valid)return d;
    Instance inst=LoadInstance(s.instance);float3 local=TransformPoint(inst.inverse,s.position);float3 previous=TransformPoint(inst.previous,local);
    d.positionDepth=float4(s.position,s.depth);d.normalRoughness=float4(s.normal,s.roughness);d.geometricMetallic=float4(s.geometric,s.metallic);
    d.albedoOpacity=float4(s.type==2?s.base:s.diffuse,s.opacity);d.specularDistance=float4(s.f0,s.hitT);
    d.motion=float4(Project(previous,true)-Project(s.position,false),dot(previous-g.previousEye.xyz,g.previousForward.xyz),s.front?1:0);d.emission=float4(s.emission,0);
    d.base=float4(s.base,s.ior);d.f90=float4(s.f90,s.diffuseMax?1:0);d.diffuseF0=float4(s.diffuseF0,s.diffuseF90.x);
    d.viewPath=float4(s.view,0);d.virtualNormal=float4(s.normal,0);d.throughput=1;d.specularMotion=d.motion;return d;
}
Surface UnpackSurface(SurfaceData d,bool previous=false) {
    Surface s=EmptySurface();if(d.identity.w==0)return s;
    s.valid=true;s.instance=d.identity.x;s.primitive=d.identity.y;s.material=d.identity.z;
    s.position=d.positionDepth.xyz;s.depth=d.positionDepth.w;s.normal=d.normalRoughness.xyz;s.roughness=d.normalRoughness.w;s.geometric=d.geometricMetallic.xyz;s.metallic=d.geometricMetallic.w;
    s.diffuse=d.albedoOpacity.xyz;s.base=d.base.xyz;s.opacity=d.albedoOpacity.w;s.f0=d.specularDistance.xyz;s.hitT=d.specularDistance.w;s.front=d.motion.w!=0;s.emission=d.emission.xyz;
    s.view=d.viewPath.xyz;s.psrDepth=(uint)d.viewPath.w;s.psrChain=asuint(d.virtualNormal.w);Material m=LoadMaterial(s.material);s.type=(uint)m.pbr.w;s.ior=max(1,m.optics.x);
    s.diffuseMax=d.f90.w!=0;s.f90=d.f90.xyz;s.diffuseF0=d.diffuseF0.xyz;s.diffuseF90=d.diffuseF0.www;return s;
}
SurfaceData ReadSurface(uint2 p,bool previous){if(previous){StructuredBuffer<SurfaceData> b=ResourceDescriptorHeap[g.guides.y];return b[p.y*g.size.x+p.x];}RWStructuredBuffer<SurfaceData> b=ResourceDescriptorHeap[g.guides.x];return b[p.y*g.size.x+p.x];}
float3 Environment(float3 world) {
    if(!g.lighting.z)return g.environment.xyz;
    float sn,cs;sincos(-g.environmentInfo.x,sn,cs);float3 d=float3(cs*world.x+sn*world.z,world.y,-sn*world.x+cs*world.z);
    Texture2D<float4> tex=ResourceDescriptorHeap[g.lighting.z];SamplerState sampler=SamplerDescriptorHeap[2];
    return tex.SampleLevel(sampler,float2(atan2(d.z,d.x)/(2*PI)+.5,acos(clamp(d.y,-1,1))/PI),0).xyz*g.environment.xyz;
}
#endif
