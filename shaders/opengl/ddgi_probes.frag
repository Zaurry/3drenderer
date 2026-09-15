#version 450 core
in vec3 v_color;
layout(location=0) out vec4 out_color;
void main(){vec2 p=gl_PointCoord*2-1;float r=dot(p,p);if(r>1)discard;out_color=vec4(v_color*(.4+.6*sqrt(1-r)),1);}
