#version 450 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;
layout(location = 3) in vec4 a_tangent;
layout(location = 4) in vec2 a_uv1;
layout(location = 5) in vec4 a_color;

uniform mat4 u_view_projection;

out VS_OUT {
    vec3 world_position;
    vec3 normal;
    vec2 uv;
    vec2 uv1;
    vec4 tangent;
    vec4 color;
} vertex_out;

void main() {
    vertex_out.world_position = a_position;
    vertex_out.normal = a_normal;
    vertex_out.uv = a_uv;
    vertex_out.uv1 = a_uv1;
    vertex_out.tangent = a_tangent;
    vertex_out.color = a_color;
    gl_Position = u_view_projection * vec4(a_position, 1.0);
}
