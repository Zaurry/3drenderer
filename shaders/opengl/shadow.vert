#version 450 core
layout(location = 0) in vec3 a_position;
layout(location = 2) in vec2 a_uv;
layout(location = 4) in vec2 a_uv1;
layout(location = 5) in vec4 a_color;
uniform mat4 u_light_view_projection;
out vec3 v_world_position;
out vec2 v_uv;
out vec2 v_uv1;
out float v_vertex_alpha;
void main() {
    v_world_position = a_position;
    v_uv = a_uv;
    v_uv1 = a_uv1;
    v_vertex_alpha = a_color.a;
    gl_Position = u_light_view_projection * vec4(a_position, 1.0);
}
