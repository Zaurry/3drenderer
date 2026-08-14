#version 450 core
out vec2 v_ndc;
void main() {
    vec2 positions[3] = vec2[3](vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    vec2 position = positions[gl_VertexID];
    v_ndc = position;
    gl_Position = vec4(position, 0.999999, 1.0);
}
