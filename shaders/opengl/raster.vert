#version 450 core

layout(location = 0) in vec3 a_position;
layout(location = 1) in vec3 a_normal;
layout(location = 2) in vec2 a_uv;
layout(location = 3) in vec4 a_tangent;

uniform mat4 u_view_projection;

out VS_OUT {
    vec3 world_position;
    vec3 normal;
    vec2 uv;
    vec4 tangent;
} vertex_out;

void main() {
    vertex_out.world_position = a_position;
    vertex_out.normal = a_normal;
    vertex_out.uv = a_uv;
    vertex_out.tangent = a_tangent;
    gl_Position = u_view_projection * vec4(a_position, 1.0);
}
