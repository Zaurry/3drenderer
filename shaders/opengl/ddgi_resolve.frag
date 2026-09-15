#version 450 core
layout(binding=0) uniform sampler2D u_base;
layout(binding=1) uniform sampler2D u_ray_source;
layout(binding=2) uniform sampler2D u_depth;
layout(binding=3) uniform sampler2D u_normal;
layout(binding=4) uniform sampler2D u_geometric_normal;
layout(binding=5) uniform sampler2D u_material;
layout(binding=6) uniform sampler2D u_fresnel;
layout(binding=7) uniform sampler2D u_pbr_aux;
layout(binding=8) uniform sampler2D u_ao;
layout(binding=9) uniform sampler2D u_irradiance;
layout(binding=10) uniform sampler2D u_distance;
layout(binding=11) uniform sampler2D u_metadata;
layout(binding=12) uniform samplerCube u_environment;
uniform vec3 u_camera_position,u_camera_forward,u_camera_right,u_camera_up;
uniform vec2 u_camera_viewport;
uniform ivec3 u_counts;
uniform vec3 u_origin,u_spacing;
uniform int u_columns,u_debug,u_ao_enabled,u_ibl_enabled,u_has_environment_map;
uniform float u_normal_bias,u_view_bias,u_intensity,u_environment_intensity,u_environment_rotation;
uniform vec3 u_environment_color;
in vec2 v_uv;
layout(location=0) out vec4 out_color;
layout(location=1) out vec4 out_ray;
const float PI=3.141592653589793;
vec2 signNZ(vec2 v){return vec2(v.x>=0?1:-1,v.y>=0?1:-1);}
vec2 octEncode(vec3 d){d/=max(dot(abs(d),vec3(1)),1e-20);return d.z<0?(1-abs(d.yx))*signNZ(d.xy):d.xy;}
vec2 atlasUV(int index,int n,vec3 direction,ivec2 size){
    vec2 base=vec2(index%u_columns,index/u_columns)*float(n+2);
    return (base+1+(octEncode(direction)*.5+.5)*float(n))/vec2(size);
}
vec3 envIrradiance(vec3 normal){
    if(u_ibl_enabled==0)return vec3(0);
    if(u_has_environment_map==0)return PI*u_environment_color*u_environment_intensity;
    vec3 tangent=normalize(cross(abs(normal.z)<.999?vec3(0,0,1):vec3(0,1,0),normal));
    vec3 bitangent=cross(normal,tangent),sum=vec3(0);
    float c=cos(-u_environment_rotation),s=sin(-u_environment_rotation);
    for(int i=0;i<32;++i){float r=sqrt((float(i)+.5)/32),phi=float(i)*2.39996323;
        vec3 d=tangent*(r*cos(phi))+bitangent*(r*sin(phi))+normal*sqrt(1-r*r);
        d=vec3(c*d.x+s*d.z,d.y,-s*d.x+c*d.z);
        sum+=textureLod(u_environment,d,2).rgb;
    }
    return sum*(PI/32)*u_environment_color*u_environment_intensity;
}
void main(){
    ivec2 pixel=ivec2(gl_FragCoord.xy);
    vec3 base=texelFetch(u_base,pixel,0).rgb,source=texelFetch(u_ray_source,pixel,0).rgb;
    out_color=vec4(base,1);out_ray=vec4(source,1);
    if(u_debug==4){out_color=vec4(texture(u_irradiance,v_uv).rgb/PI,1);return;}
    if(u_debug==5){float d=texture(u_distance,v_uv).r;out_color=vec4(vec3(d/(d+max(u_spacing.x,max(u_spacing.y,u_spacing.z)))),1);return;}
    float depth=texelFetch(u_depth,pixel,0).r;if(depth<=0)return;
    vec2 xy=(v_uv*2-1)*u_camera_viewport*.5;
    vec3 position=u_camera_position+depth*(u_camera_forward+xy.x*u_camera_right+xy.y*u_camera_up);
    vec3 view=normalize(u_camera_position-position);
    vec3 vn=texelFetch(u_normal,pixel,0).xyz;
    vec3 normal=normalize(vn.x*u_camera_right+vn.y*u_camera_up-vn.z*u_camera_forward);
    vec3 geometric=normalize(texelFetch(u_geometric_normal,pixel,0).xyz);
    float cell=min(u_spacing.x,min(u_spacing.y,u_spacing.z));
    vec3 biased=position+cell*(geometric*u_normal_bias+view*u_view_bias);
    vec3 rawGrid=(biased-u_origin)/u_spacing;
    vec3 outside=max(max(-rawGrid,rawGrid-vec3(u_counts-1)),vec3(0));
    float volumeWeight=1-clamp(max(outside.x,max(outside.y,outside.z)),0,1);
    vec3 grid=clamp(rawGrid,vec3(0),vec3(u_counts-1));
    ivec3 cellIndex=min(ivec3(floor(grid)),u_counts-2);vec3 fraction=grid-vec3(cellIndex);
    vec3 irradiance=vec3(0);float total=0,age=0,validCount=0;
    if(volumeWeight>0)for(int corner=0;corner<8;++corner){
        ivec3 bits=ivec3(corner&1,(corner>>1)&1,(corner>>2)&1),coord=cellIndex+bits;
        int index=coord.x+u_counts.x*(coord.y+u_counts.y*coord.z);
        vec4 metadata=texelFetch(u_metadata,ivec2(index,0),0);
        if(metadata.w<.5)continue;
        validCount+=1;
        vec3 p=u_origin+vec3(coord)*u_spacing+metadata.xyz;
        vec3 delta=biased-p;float d=max(length(delta),1e-10);vec3 direction=delta/d;
        vec2 moments=texture(u_distance,atlasUV(index,16,direction,textureSize(u_distance,0))).rg;
        float variance=max(moments.y-moments.x*moments.x,cell*cell*1e-6);
        float excess=max(d-moments.x,0),visibility=variance/(variance+excess*excess);visibility=visibility*visibility*visibility;
        vec3 weights=mix(1-fraction,fraction,vec3(bits));float facing=d<cell*1e-5?1:max(0,-dot(geometric,direction));
        float weight=weights.x*weights.y*weights.z*facing*facing*visibility;
        if(visibility<.2)weight*=visibility*visibility/.04;
        irradiance+=texture(u_irradiance,atlasUV(index,8,normal,textureSize(u_irradiance,0))).rgb*weight;
        age+=texelFetch(u_metadata,ivec2(index,1),0).z*weight;total+=weight;
    }
    irradiance=total>1e-8?irradiance/total:vec3(0);
    // Only the outside of the volume fades to IBL. Occlusion inside stays dark.
    if(volumeWeight<1)irradiance=mix(envIrradiance(normal),irradiance,volumeWeight);
    vec4 material=texelFetch(u_material,pixel,0),f=texelFetch(u_fresnel,pixel,0);
    vec3 fresnel=f.rgb+(vec3(f.a)-f.rgb)*pow(1-clamp(dot(normal,view),0,1),5);
    vec3 diffuseWeight=(int(material.a+.5)&1)!=0?vec3(1-max(fresnel.r,max(fresnel.g,fresnel.b))):1-fresnel;
    float ao=texelFetch(u_pbr_aux,pixel,0).a*(u_ao_enabled!=0?texelFetch(u_ao,pixel,0).a:1);
    vec3 indirect=max(irradiance*material.rgb*diffuseWeight*(ao*u_intensity/PI),vec3(0));
    out_color=vec4(base+indirect,1);out_ray=vec4(source+indirect,1);
    if(u_debug==1)out_color=vec4(indirect,1);
    if(u_debug==2)out_color=vec4(1-validCount/8,validCount/8,0,1);
    if(u_debug==3)out_color=vec4(vec3(total>1e-8?clamp(age/total/32,0,1):1),1);
}
