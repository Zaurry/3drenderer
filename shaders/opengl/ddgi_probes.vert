#version 450 core
layout(binding=0) uniform sampler2D u_metadata;
uniform ivec3 u_counts;
uniform vec3 u_origin,u_spacing;
uniform mat4 u_view_projection;
uniform float u_screen_height;
out vec3 v_color;
void main(){
    int index=gl_VertexID;vec4 data=texelFetch(u_metadata,ivec2(index,0),0);
    ivec3 grid=ivec3(index%u_counts.x,(index/u_counts.x)%u_counts.y,index/(u_counts.x*u_counts.y));
    gl_Position=u_view_projection*vec4(u_origin+vec3(grid)*u_spacing+data.xyz,1);
    gl_PointSize=clamp(min(u_spacing.x,min(u_spacing.y,u_spacing.z))*.12*u_screen_height/max(gl_Position.w,.001),2,50);
    v_color=data.w>.5?vec3(.1,1,.25):(data.w<0?vec3(1,.12,.05):vec3(1,.8,.1));
}
